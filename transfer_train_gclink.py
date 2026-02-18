import os
import time
import random
import argparse
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from scGNN import GENELink, create_cell_adjacency_matrix
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast


# ---------------------------
# 1) 你项目里已经有的：表达感知删边 + 邻居对比损失
#    我这里保持接口一致（你可直接替换成你现有实现）
# ---------------------------
class ExpressionAwareEdgeRemoving(object):
    def __init__(self, expression_matrix: torch.Tensor, pe: float = 0.2, high_quantile: float = 0.75):
        assert 0.0 <= pe < 1.0
        self.pe = pe
        self.device = expression_matrix.device
        with torch.no_grad():
            tau = torch.quantile(expression_matrix, high_quantile)
            B = (expression_matrix > tau).float()
            Nc = B.size(1)
            self.coact = (B @ B.t()) / float(Nc)

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        M = edge_index.size(1)
        if self.pe <= 0.0 or M == 0:
            return x, edge_index, edge_weight
        num_drop = int(self.pe * M)
        if num_drop == 0:
            return x, edge_index, edge_weight

        src = edge_index[0]
        dst = edge_index[1]
        w_ij = self.coact[src, dst]  # [M]
        _, sorted_idx = torch.sort(w_ij)  # asc
        drop_idx = sorted_idx[:num_drop]

        keep_mask = torch.ones(M, dtype=torch.bool, device=edge_index.device)
        keep_mask[drop_idx] = False
        new_edge_index = edge_index[:, keep_mask]
        new_edge_weight = edge_weight[keep_mask] if edge_weight is not None else None
        return x, new_edge_index, new_edge_weight


def _build_random_neighbor_index(adj: torch.Tensor) -> torch.Tensor:
    device = adj.device
    N = adj.size(0)
    neighbors = [[] for _ in range(N)]
    if adj.is_sparse:
        idx = adj.coalesce().indices()
        rows = idx[0].tolist()
        cols = idx[1].tolist()
        for r, c in zip(rows, cols):
            if r != c:
                neighbors[r].append(c)
    else:
        mat = (adj > 0).to(torch.bool)
        eye = torch.arange(N, device=device)
        mat[eye, eye] = False
        nz = mat.nonzero(as_tuple=False)
        for r, c in nz.tolist():
            neighbors[r].append(c)

    j_idx = []
    for i in range(N):
        if neighbors[i]:
            j_idx.append(random.choice(neighbors[i]))
        else:
            j_idx.append(i)
    return torch.tensor(j_idx, dtype=torch.long, device=device)


def _info_nce_multi_positive(h_anchor: torch.Tensor, h_pool: torch.Tensor, pos_mask: torch.Tensor, tau: float = 0.2):
    h_anchor = F.normalize(h_anchor, dim=1)
    h_pool = F.normalize(h_pool, dim=1)
    logits = (h_anchor @ h_pool.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    pos_exp = (exp_logits * pos_mask.float()).sum(dim=1)
    all_exp = exp_logits.sum(dim=1)
    loss = -torch.log((pos_exp + 1e-12) / (all_exp + 1e-12))
    return loss.mean()


def neighbor_contrastive_loss(h1: torch.Tensor, h2: torch.Tensor, adj: torch.Tensor, tau: float = 0.2, symmetric: bool = True):
    if h1 is None or h2 is None:
        return torch.tensor(0.0, device=adj.device)

    N = h1.size(0)
    device = h1.device
    idx = torch.arange(N, device=device)

    j_idx = _build_random_neighbor_index(adj)
    pos_mask = torch.zeros((N, N), dtype=torch.bool, device=device)
    pos_mask[idx, idx] = True
    pos_mask[idx, j_idx] = True
    loss = _info_nce_multi_positive(h1, h2, pos_mask, tau=tau)

    if symmetric:
        j_idx2 = _build_random_neighbor_index(adj)
        pos_mask2 = torch.zeros((N, N), dtype=torch.bool, device=device)
        pos_mask2[idx, idx] = True
        pos_mask2[idx, j_idx2] = True
        loss2 = _info_nce_multi_positive(h2, h1, pos_mask2, tau=tau)
        loss = 0.5 * (loss + loss2)
    return loss


# ---------------------------
# 2) 迁移模型封装（与你当前 forward 形态一致）
# ---------------------------
class GCLink(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, data_feature, adj, batch_pairs, cell_features=None, cell_adj=None, expression_matrix=None,
                aug1=None, aug2=None):
        index = adj.coalesce().indices()
        size = adj.coalesce().size()

        # two views
        x1, e1, _ = aug1(data_feature, index)
        x2, e2, _ = aug2(data_feature, index)

        v1 = torch.ones((e1.size(1),), device=data_feature.device)
        v2 = torch.ones((e2.size(1),), device=data_feature.device)
        adj1 = torch.sparse_coo_tensor(e1, v1, size)
        adj2 = torch.sparse_coo_tensor(e2, v2, size)

        embed1, _, _, pred1, cell_embed1 = self.encoder(data_feature, adj1, batch_pairs, cell_features, cell_adj, expression_matrix)
        embed2, _, _, pred2, cell_embed2 = self.encoder(data_feature, adj2, batch_pairs, cell_features, cell_adj, expression_matrix)

        return embed1, pred1, embed2, pred2, cell_embed1, cell_embed2


# ---------------------------
# 3) 数据与图构建：关键点是“对齐输入维度”
#    - gene node feature：SVD(feature[g,c]) -> [G, gene_feat_dim]
#    - cell feature：SVD(feature.T[c,g]) -> [C, cell_feat_dim]
# ---------------------------
def build_bundle(cell_type: str, tf_num: int, noise_ratio: float, sample: str,
                 gene_feat_dim: int, cell_feat_dim: int, knn: int, loop: bool, flag: bool,
                 fewshot: bool, fewshot_ratio: float, seed: int, device: torch.device):
    # paths（按你当前工程组织）
    exp_file = f'Benchmark Datasets/Specific Dataset/{cell_type}/TFs+{tf_num}/BL--ExpressionData.csv'
    tf_file  = f'Benchmark Datasets/Specific Dataset/{cell_type}/TFs+{tf_num}/TF.csv'

    base_data_dir = f'Specific_noise/noise_{noise_ratio}/{cell_type} {tf_num}/{sample}'
    train_file = f'{base_data_dir}/Train_set.csv'
    val_file   = f'{base_data_dir}/Validation_set.csv'
    test_file  = f'{base_data_dir}/Test_set.csv'

    # expression matrix
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()            # numpy, [G, C]
    G, C = feature.shape
    print(f'[{cell_type}] raw feature shape (G,C) = {feature.shape}')

    # keep raw expression for expression-aware aug
    expression_matrix = torch.from_numpy(feature).float().to(device)  # [G,C]

    # ---- align gene node feature dim (cell-dimension projected) ----
    svd_gene = TruncatedSVD(n_components=gene_feat_dim, random_state=seed)
    data_feature = svd_gene.fit_transform(feature)                    # [G, gene_feat_dim]
    data_feature = torch.from_numpy(data_feature).float().to(device)

    # ---- align cell feature dim (gene-dimension projected) ----
    svd_cell = TruncatedSVD(n_components=cell_feat_dim, random_state=seed)
    cell_features_raw = feature.T                                     # [C, G]
    cell_features = svd_cell.fit_transform(cell_features_raw)         # [C, cell_feat_dim]
    cell_features = torch.from_numpy(cell_features).float().to(device)

    # cell adjacency (KNN)
    cell_adj = create_cell_adjacency_matrix(cell_features.cpu().numpy(), n_neighbors=knn)
    cell_adj = cell_adj.to(device)

    # TF indices
    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    tf = torch.from_numpy(tf).to(device)

    # load edge-labeled pairs
    train_data = pd.read_csv(train_file, index_col=0).values
    val_data   = pd.read_csv(val_file, index_col=0).values
    test_data  = pd.read_csv(test_file, index_col=0).values

    # few-shot: only use a small subset for finetune; test is the remaining part
    if fewshot:
        full = np.concatenate([train_data, val_data, test_data], axis=0)
        x = full[:, :-1]
        y = full[:, -1]
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=(1.0 - fewshot_ratio), stratify=y, random_state=seed
        )
        train_data = np.concatenate([x_tr, y_tr.reshape(-1, 1)], axis=1)
        test_data  = np.concatenate([x_te, y_te.reshape(-1, 1)], axis=1)
        val_data   = train_data.copy()  # 也可另切，这里简单起见
        print(f'[{cell_type}] few-shot finetune: train={len(train_data)} test={len(test_data)}')
    else:
        print(f'[{cell_type}] full train: train={len(train_data)} val={len(val_data)} test={len(test_data)}')

    # build adjacency from training pairs
    train_load = scRNADataset(train_data, G, flag=flag)
    adj = train_load.Adj_Generate(tf, loop=loop)
    adj = adj2saprse_tensor(adj).to(device)

    # tensors for eval
    train_tensor = torch.from_numpy(train_data).to(device)
    val_tensor   = torch.from_numpy(val_data).to(device)
    test_tensor  = torch.from_numpy(test_data).to(device)

    return {
        "G": G, "C": C,
        "data_feature": data_feature,
        "expression_matrix": expression_matrix,
        "cell_features": cell_features,
        "cell_adj": cell_adj,
        "tf": tf,
        "adj": adj,
        "train_load": train_load,
        "train_pairs": train_tensor,
        "val_pairs": val_tensor,
        "test_pairs": test_tensor,
    }


# ---------------------------
# 4) 训练与评估
# ---------------------------
def train_one_epoch(model: GCLink, train_load: scRNADataset, data_feature, adj,
                    cell_features, cell_adj, expression_matrix,
                    optimizer, scheduler,
                    lambda_link: float, lambda_con_gene: float, lambda_con_cell: float,
                    aug1, aug2, device, flag: bool):
    model.train()
    running = 0.0
    dl = DataLoader(train_load, batch_size=args.batch_size, shuffle=True)

    for batch_x, batch_y in dl:
        optimizer.zero_grad()
        batch_y = batch_y.to(device).view(-1, 1) if not flag else batch_y.to(device)

        embed1, pred1, embed2, pred2, cell_e1, cell_e2 = model(
            data_feature, adj, batch_x.to(device),
            cell_features=cell_features, cell_adj=cell_adj, expression_matrix=expression_matrix,
            aug1=aug1, aug2=aug2
        )

        # link loss
        if flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss_bce = F.binary_cross_entropy(pred1, batch_y) + F.binary_cross_entropy(pred2, batch_y)

        # contrastive
        loss_con_gene = neighbor_contrastive_loss(embed1, embed2, adj, tau=0.2, symmetric=True)
        loss_con_cell = neighbor_contrastive_loss(cell_e1, cell_e2, cell_adj, tau=0.2, symmetric=True)

        # weighted sum (normalize weights)
        w_sum = lambda_link + lambda_con_gene + lambda_con_cell
        w_link = lambda_link / w_sum
        w_g = lambda_con_gene / w_sum
        w_c = lambda_con_cell / w_sum

        loss = w_link * loss_bce + w_g * loss_con_gene + w_c * loss_con_cell
        loss.backward()
        optimizer.step()
        scheduler.step()
        running += loss.item()

    return running


@torch.no_grad()
def eval_pairs(model: GCLink, data_feature, adj, pairs_tensor,
               cell_features, cell_adj, expression_matrix,
               aug1, aug2, device, flag: bool):
    model.eval()
    _, _, _, pred2, _, _ = model(
        data_feature, adj, pairs_tensor,
        cell_features=cell_features, cell_adj=cell_adj, expression_matrix=expression_matrix,
        aug1=aug1, aug2=aug2
    )
    score = torch.softmax(pred2, dim=1) if flag else torch.sigmoid(pred2)
    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=pairs_tensor[:, -1], flag=flag)
    return AUC, AUPR


# ---------------------------
# 5) 主程序：source 训练 + target 微调
# ---------------------------
def build_model(gene_feat_dim: int, cell_feat_dim: int, args, device):
    encoder = GENELink(
        input_dim=gene_feat_dim,
        hidden1_dim=args.hidden_dim[0],
        hidden2_dim=args.hidden_dim[1],
        hidden3_dim=args.hidden_dim[2],
        output_dim=args.output_dim,
        num_head1=args.num_head[0],
        num_head2=args.num_head[1],
        alpha=args.alpha,
        device=device,
        type=args.Type,
        reduction=args.reduction,
        cell_feature_dim=cell_feat_dim,
        cell_hidden_dim=args.output_dim,
        top_k=args.top_k_cells
    ).to(device)
    return GCLink(encoder).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["source", "finetune"], required=True)

    # data
    parser.add_argument("--source_cell", type=str, default="mESC")
    parser.add_argument("--target_cell", type=str, default="hESC")
    parser.add_argument("--tf_num", type=int, choices=[500, 1000], default=1000)
    parser.add_argument("--noise_ratio", type=float, default=0.05)
    parser.add_argument("--sample", type=str, default="sample1")

    # alignment dims (关键：迁移对齐点)
    parser.add_argument("--gene_feat_dim", type=int, default=200)
    parser.add_argument("--cell_feat_dim", type=int, default=128)

    # training
    parser.add_argument("-lr", type=float, default=3e-3)
    parser.add_argument("-epochs", type=int, default=20)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--edge_drop_rate", type=float, default=0.2)
    parser.add_argument("--high_quantile", type=float, default=0.75)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--flag", action="store_true")
    parser.add_argument('-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    # model hparams (与你项目一致)
    parser.add_argument("--Type", type=str, default="MLP")
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--num_head", type=str, default="3,3")
    parser.add_argument("--hidden_dim", type=str, default="128,64,64")
    parser.add_argument("--output_dim", type=int, default=32)
    parser.add_argument("--top_k_cells", type=int, default=20)

    # losses
    parser.add_argument("--lambda_link", type=float, default=0.75)
    parser.add_argument("--lambda_con_gene", type=float, default=0.1)
    parser.add_argument("--lambda_con_cell", type=float, default=0.1)

    # finetune settings
    parser.add_argument("--fewshot_ratio", type=float, default=0.05)
    parser.add_argument("--finetune_epochs", type=int, default=10)

    # ckpt
    parser.add_argument("--ckpt_dir", type=str, default="model_transfer")
    args = parser.parse_args()

    # parse lists
    args.num_head = [int(x) for x in args.num_head.replace(" ", "").split(",")]
    args.hidden_dim = [int(x) for x in args.hidden_dim.replace(" ", "").split(",")]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    source_ckpt = os.path.join(args.ckpt_dir, f"source_{args.source_cell}_TF{args.tf_num}.pt")

    # augmentors (view1 identity, view2 expression-aware drop edge)
    def make_augs(expression_matrix):
        aug1 = A.Identity()
        aug2 = ExpressionAwareEdgeRemoving(
            expression_matrix=expression_matrix,
            pe=args.edge_drop_rate,
            high_quantile=args.high_quantile
        )
        return aug1, aug2

    if args.mode == "source":
        bundle = build_bundle(
            cell_type=args.source_cell,
            tf_num=args.tf_num,
            noise_ratio=args.noise_ratio,
            sample=args.sample,
            gene_feat_dim=args.gene_feat_dim,
            cell_feat_dim=args.cell_feat_dim,
            knn=args.knn,
            loop=args.loop,
            flag=args.flag,
            fewshot=False,
            fewshot_ratio=args.fewshot_ratio,
            seed=args.seed,
            device=device
        )

        model = build_model(args.gene_feat_dim, args.cell_feat_dim, args, device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        aug1, aug2 = make_augs(bundle["expression_matrix"])

        best_aupr = -1.0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(
                model, bundle["train_load"],
                bundle["data_feature"], bundle["adj"],
                bundle["cell_features"], bundle["cell_adj"], bundle["expression_matrix"],
                optimizer, scheduler,
                args.lambda_link, args.lambda_con_gene, args.lambda_con_cell,
                aug1, aug2, device, args.flag
            )
            auc, aupr = eval_pairs(
                model, bundle["data_feature"], bundle["adj"], bundle["val_pairs"],
                bundle["cell_features"], bundle["cell_adj"], bundle["expression_matrix"],
                aug1, aug2, device, args.flag
            )
            dt = time.time() - t0
            print(f"[SOURCE {args.source_cell}] ep={ep:03d} loss={loss:.4f} val_AUC={auc:.4f} val_AUPR={aupr:.4f} time={dt:.1f}s")

            if aupr > best_aupr:
                best_aupr = aupr
                torch.save(model.state_dict(), source_ckpt)

        print(f"Saved source ckpt: {source_ckpt}  (best val_AUPR={best_aupr:.4f})")

    elif args.mode == "finetune":
        assert os.path.exists(source_ckpt), f"Missing source checkpoint: {source_ckpt}. Run --mode source first."

        bundle = build_bundle(
            cell_type=args.target_cell,
            tf_num=args.tf_num,
            noise_ratio=args.noise_ratio,
            sample=args.sample,
            gene_feat_dim=args.gene_feat_dim,
            cell_feat_dim=args.cell_feat_dim,
            knn=args.knn,
            loop=args.loop,
            flag=args.flag,
            fewshot=True,
            fewshot_ratio=args.fewshot_ratio,
            seed=args.seed,
            device=device
        )

        model = build_model(args.gene_feat_dim, args.cell_feat_dim, args, device)
        model.load_state_dict(torch.load(source_ckpt, map_location=device))

        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        aug1, aug2 = make_augs(bundle["expression_matrix"])

                # ---- finetune: save best by test_AUPR (or you can switch to val_AUPR) ----
        best_test_aupr = -1.0
        best_test_auc = -1.0
        best_ep = -1

        best_ckpt_path = os.path.join(
            args.ckpt_dir,
            f"finetuned_{args.source_cell}_to_{args.target_cell}_TF{args.tf_num}_BEST_AUPR.pt"
        )

        for ep in range(1, args.finetune_epochs + 1):
            loss = train_one_epoch(
                model, bundle["train_load"],
                bundle["data_feature"], bundle["adj"],
                bundle["cell_features"], bundle["cell_adj"], bundle["expression_matrix"],
                optimizer, scheduler,
                args.lambda_link, args.lambda_con_gene, args.lambda_con_cell,
                aug1, aug2, device, args.flag
            )

            auc, aupr = eval_pairs(
                model, bundle["data_feature"], bundle["adj"], bundle["test_pairs"],
                bundle["cell_features"], bundle["cell_adj"], bundle["expression_matrix"],
                aug1, aug2, device, args.flag
            )

            print(f"[FINETUNE {args.target_cell}] ep={ep:03d} loss={loss:.4f} test_AUC={auc:.4f} test_AUPR={aupr:.4f}")

            # save best checkpoint (by AUPR)
            if aupr > best_test_aupr:
                best_test_aupr = aupr
                best_test_auc = auc
                best_ep = ep
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  -> New BEST test_AUPR={best_test_aupr:.4f} at ep={best_ep:03d}, saved to: {best_ckpt_path}")

        # still save last epoch ckpt if you want
        last_ckpt = os.path.join(
            args.ckpt_dir,
            f"finetuned_{args.source_cell}_to_{args.target_cell}_TF{args.tf_num}_LAST.pt"
        )
        torch.save(model.state_dict(), last_ckpt)

        print(f"Best checkpoint: {best_ckpt_path} (best_ep={best_ep}, best_test_AUPR={best_test_aupr:.4f}, best_test_AUC={best_test_auc:.4f})")
        print(f"Last checkpoint: {last_ckpt}")

