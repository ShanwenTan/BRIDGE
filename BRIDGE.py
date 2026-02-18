import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import time
import math
import random
import argparse
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast

from scGNN import GENELink, create_cell_adjacency_matrix
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation


NET_PRESETS: Dict[str, Dict[str, Any]] = {
    "specific": {"epochs": 20, "knn": 20, "Type": "MLP"},
    "non_specific": {"epochs": 20, "knn": 40, "Type": "cosine"},
    "string": {"epochs": 20, "knn": 20, "Type": "cosine"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, choices=["specific", "non_specific", "string"])
    parser.add_argument("--cell_type", type=str, required=True)
    parser.add_argument("--sample", type=str, default="sample1")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--num_head", type=str, default="3,3")
    parser.add_argument("--hidden_dim", type=str, default="128,64,64")
    parser.add_argument("--output_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--reduction", type=str, default="concate")
    parser.add_argument("--flag", action="store_true")
    parser.add_argument("--top_k_cells", type=int, default=20)
    parser.add_argument("--lambda_link", type=float, default=0.8)
    parser.add_argument("--lambda_con_gene", type=float, default=0.1)
    parser.add_argument("--lambda_con_cell", type=float, default=0.1)
    parser.add_argument("--edge_drop_rate", type=float, default=0.2)
    parser.add_argument("--high_quantile", type=float, default=0.8)
    parser.add_argument("--tf_num", type=int, choices=[500, 1000], default=1000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--knn", type=int, default=None)
    parser.add_argument("--Type", type=str, default=None)
    parser.add_argument("--use_preset", action="store_true", default=True)
    parser.add_argument("--no_preset", dest="use_preset", action="store_false")
    return parser


def apply_net_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.use_preset:
        if args.epochs is None:
            args.epochs = 20
        if args.knn is None:
            args.knn = 20
        if args.Type is None:
            args.Type = "cosine"
        return args
    preset = NET_PRESETS[args.net]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.knn is None:
        args.knn = preset["knn"]
    if args.Type is None:
        args.Type = preset["Type"]
    return args


def parse_and_finalize_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args.hidden_dim = [int(x) for x in args.hidden_dim.strip("[]").split(",")]
    args.num_head = [int(x) for x in args.num_head.strip("[]").split(",")]
    args = apply_net_preset(args)
    return args


args = parse_and_finalize_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_epoch_gate_all(model, data_feature, adj, cell_features, cell_adj, expression_matrix):
    model.eval()
    _ = model.encoder.encode(
        x=data_feature,
        adj=adj,
        cell_features=cell_features,
        cell_adj=cell_adj,
        expression_matrix=expression_matrix,
    )
    gammas_g, gammas_c = [], []
    layers = []
    if hasattr(model.encoder, "hetero_layer1"):
        layers.append(model.encoder.hetero_layer1)
    if hasattr(model.encoder, "hetero_layer2"):
        layers.append(model.encoder.hetero_layer2)
    for layer in layers:
        if not hasattr(layer, "heads"):
            continue
        for head in layer.heads:
            if hasattr(head, "_gamma_g_mean") and hasattr(head, "_gamma_c_mean"):
                gammas_g.append(head._gamma_g_mean)
                gammas_c.append(head._gamma_c_mean)
    if len(gammas_g) == 0:
        return float("nan"), float("nan")
    gamma_g_mean = torch.stack(gammas_g).mean().item()
    gamma_c_mean = torch.stack(gammas_c).mean().item()
    return gamma_g_mean, gamma_c_mean


def resolve_paths(args):
    tf_num = args.tf_num
    if args.net == "specific":
        exp_file = f"Benchmark Datasets/Specific Dataset/{args.cell_type}/TFs+{tf_num}/BL--ExpressionData.csv"
        tf_file = f"Benchmark Datasets/Specific Dataset/{args.cell_type}/TFs+{tf_num}/TF.csv"
        base_dir = f"Data/Specific/{args.cell_type} {tf_num}/{args.sample}"
    elif args.net == "non_specific":
        exp_file = f"Benchmark Datasets/Non-Specific Dataset/{args.cell_type}/TFs+{tf_num}/BL--ExpressionData.csv"
        tf_file = f"Benchmark Datasets/Non-Specific Dataset/{args.cell_type}/TFs+{tf_num}/TF.csv"
        base_dir = f"Data/Non-Specific/{args.cell_type} {tf_num}/{args.sample}"
    elif args.net == "string":
        exp_file = f"Benchmark Datasets/STRING Dataset/{args.cell_type}/TFs+{tf_num}/BL--ExpressionData.csv"
        tf_file = f"Benchmark Datasets/STRING Dataset/{args.cell_type}/TFs+{tf_num}/TF.csv"
        base_dir = f"Data/STRING/{args.cell_type} {tf_num}/{args.sample}"
    else:
        raise ValueError(f"Unknown net: {args.net}")
    train_file = f"{base_dir}/Train_set.csv"
    val_file = f"{base_dir}/Validation_set.csv"
    test_file = f"{base_dir}/Test_set.csv"
    return exp_file, tf_file, train_file, val_file, test_file


def _build_random_neighbor_index(adj: torch.Tensor, allow_self_fallback: bool = True) -> torch.Tensor:
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
        eye_idx = torch.arange(N, device=adj.device)
        mat[eye_idx, eye_idx] = False
        nz = mat.nonzero(as_tuple=False)
        for r, c in nz.tolist():
            neighbors[r].append(c)
    j_idx = []
    for i in range(N):
        if neighbors[i]:
            j_idx.append(random.choice(neighbors[i]))
        else:
            j_idx.append(i if allow_self_fallback else i)
    return torch.tensor(j_idx, dtype=torch.long, device=adj.device)


def _info_nce_multi_positive(h_anchor: torch.Tensor, h_pool: torch.Tensor, pos_mask: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    h_anchor = F.normalize(h_anchor, dim=1)
    h_pool = F.normalize(h_pool, dim=1)
    logits = torch.matmul(h_anchor, h_pool.t()) / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    pos_exp = (exp_logits * pos_mask.float()).sum(dim=1)
    all_exp = exp_logits.sum(dim=1)
    loss = -torch.log((pos_exp + 1e-12) / (all_exp + 1e-12))
    return loss.mean()


def neighbor_contrastive_loss(h1: torch.Tensor, h2: torch.Tensor, adj: torch.Tensor, tau: float = 0.2, symmetric: bool = True) -> torch.Tensor:
    if (h1 is None) or (h2 is None):
        return torch.tensor(0.0, device=adj.device)
    N = h1.size(0)
    idx = torch.arange(N, device=h1.device)
    j_idx = _build_random_neighbor_index(adj)
    pos_mask = torch.zeros((N, N), dtype=torch.bool, device=h1.device)
    pos_mask[idx, idx] = True
    pos_mask[idx, j_idx] = True
    loss = _info_nce_multi_positive(h1, h2, pos_mask, tau=tau)
    if symmetric:
        j_idx_2 = _build_random_neighbor_index(adj)
        pos_mask_2 = torch.zeros((N, N), dtype=torch.bool, device=h1.device)
        pos_mask_2[idx, idx] = True
        pos_mask_2[idx, j_idx_2] = True
        loss_rev = _info_nce_multi_positive(h2, h1, pos_mask_2, tau=tau)
        loss = 0.5 * (loss + loss_rev)
    return loss


class ExpressionAwareEdgeRemoving(object):
    def __init__(self, expression_matrix: torch.Tensor, pe: float = 0.2, high_quantile: float = 0.8):
        assert 0.0 <= pe < 1.0
        self.pe = pe
        self.device = expression_matrix.device
        with torch.no_grad():
            tau = torch.quantile(expression_matrix, high_quantile)
            B = (expression_matrix > tau).float()
            Nc = B.size(1)
            w = (B @ B.t()) / float(Nc)
            self.coact = w

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        num_edges = edge_index.size(1)
        if self.pe <= 0.0 or num_edges == 0:
            return x, edge_index, edge_weight
        num_drop = int(self.pe * num_edges)
        if num_drop == 0:
            return x, edge_index, edge_weight
        src = edge_index[0]
        dst = edge_index[1]
        w_ij = self.coact[src, dst]
        _, sorted_idx = torch.sort(w_ij)
        drop_idx = sorted_idx[:num_drop]
        keep_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
        keep_mask[drop_idx] = False
        new_edge_index = edge_index[:, keep_mask]
        new_edge_weight = edge_weight[keep_mask] if edge_weight is not None else None
        return x, new_edge_index, new_edge_weight


class GCLink(nn.Module):
    def __init__(self, encoder):
        super(GCLink, self).__init__()
        self.encoder = encoder

    def forward(self, data_feature, adj, train_data, cell_features=None, cell_adj=None, expression_matrix=None):
        index = adj.coalesce().indices()
        size = adj.coalesce().size()
        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)
        v1 = torch.ones((edge_index1.shape[1]), device=device)
        v2 = torch.ones((edge_index2.shape[1]), device=device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)
        embed1, tf_embed1, target_embed1, pred1, cell_embed1 = self.encoder(
            data_feature, adj1, train_data, cell_features, cell_adj, expression_matrix
        )
        embed2, tf_embed2, target_embed2, pred2, cell_embed2 = self.encoder(
            data_feature, adj2, train_data, cell_features, cell_adj, expression_matrix
        )
        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2, cell_embed1, cell_embed2


def gnn_train(data_feature, adj1, adj2, gnn_model, optimizer, scheduler, cell_features, cell_adj, expression_matrix):
    pretrain_loss = 0.0
    data_loader = DataLoader(train_load, batch_size=args.batch_size, shuffle=True)
    for train_x, train_y in data_loader:
        gnn_model.train()
        optimizer.zero_grad()
        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)
        z1, train_tf1, train_target1, pred1, _ = gnn_model(data_feature, adj1, train_x, cell_features, cell_adj, expression_matrix)
        z2, train_tf2, train_target2, pred2, _ = gnn_model(data_feature, adj2, train_x, cell_features, cell_adj, expression_matrix)
        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)
        loss1 = F.binary_cross_entropy(pred1, train_y)
        loss2 = F.binary_cross_entropy(pred2, train_y)
        loss = loss1 + loss2
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        pretrain_loss += loss.item()
    return float(pretrain_loss), pred1, pred2


def pretrain(data_feature, adj, model, cell_features, cell_adj, expression_matrix, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    index = adj.coalesce().indices()
    size = adj.coalesce().size()
    for epoch in range(1, epochs + 1):
        x1, edge_index1, edge_weight1 = aug1(data_feature, index)
        x2, edge_index2, edge_weight2 = aug2(data_feature, index)
        v1 = torch.ones((edge_index1.shape[1]), device=device)
        v2 = torch.ones((edge_index2.shape[1]), device=device)
        adj1 = torch.sparse_coo_tensor(edge_index1, v1, size)
        adj2 = torch.sparse_coo_tensor(edge_index2, v2, size)
        pre_train_loss, pred1, pred2 = gnn_train(data_feature, adj1, adj2, model, optimizer, scheduler, cell_features, cell_adj, expression_matrix)
        print("Epoch:{}".format(epoch), "pre-train loss:{:.5F}".format(pre_train_loss))


def train(model, contrast_model_gene, contrast_model_cell, optimizer, scheduler, cell_features, cell_adj, expression_matrix):
    running_loss = 0.0
    data_loader = DataLoader(train_load, batch_size=args.batch_size, shuffle=True)
    for train_x, train_y in data_loader:
        model.train()
        optimizer.zero_grad()
        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)
        embed1, _, _, pred1, embed2, _, _, pred2, cell_embed1, cell_embed2 = model(data_feature, adj, train_x, cell_features, cell_adj, expression_matrix)
        con_loss_gene = neighbor_contrastive_loss(embed1, embed2, adj, tau=0.2, symmetric=True)
        con_loss_cell = neighbor_contrastive_loss(cell_embed1, cell_embed2, cell_adj, tau=0.2, symmetric=True)
        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)
        loss_BCE1 = F.binary_cross_entropy(pred1, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2, train_y)
        link_prediction_loss = loss_BCE1 + loss_BCE2
        w_sum = (args.lambda_link + args.lambda_con_gene + args.lambda_con_cell)
        w_link = args.lambda_link / w_sum
        w_gene = args.lambda_con_gene / w_sum
        w_cell = args.lambda_con_cell / w_sum
        total_loss = w_link * link_prediction_loss + w_gene * con_loss_gene + w_cell * con_loss_cell
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += total_loss.item()
    return running_loss


tf_num = args.tf_num
exp_file, tf_file, train_file, val_file, test_file = resolve_paths(args)

data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature_np = loader.exp_data()

cell_features_raw = feature_np.T
cell_adj = create_cell_adjacency_matrix(cell_features_raw, n_neighbors=args.knn).to(device)
cell_features = torch.from_numpy(cell_features_raw).float().to(device)
expression_matrix = torch.from_numpy(feature_np).float().to(device)

tf = pd.read_csv(tf_file, index_col=0)["index"].values.astype(np.int64)
tf = torch.from_numpy(tf).to(device)

feature = torch.from_numpy(feature_np).to(device)
data_feature = feature

train_data_np = pd.read_csv(train_file, index_col=0).values
test_data_np = pd.read_csv(test_file, index_col=0).values
val_data_np = pd.read_csv(val_file, index_col=0).values

train_load = scRNADataset(train_data_np, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf, loop=args.loop)
adj = adj2saprse_tensor(adj).to(device)

train_data = torch.from_numpy(train_data_np).to(device)
test_data = torch.from_numpy(test_data_np).to(device)
validation_data = torch.from_numpy(val_data_np).to(device)

contrast_model_gene = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode="L2L", intraview_negs=False).to(device)
contrast_model_cell = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode="L2L", intraview_negs=False).to(device)

encoder = GENELink(
    input_dim=feature.size()[1],
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
    cell_feature_dim=cell_features.size()[1],
    cell_hidden_dim=args.output_dim,
    top_k=args.top_k_cells,
).to(device)

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

aug1 = A.Identity()
aug2 = ExpressionAwareEdgeRemoving(expression_matrix=expression_matrix, pe=args.edge_drop_rate, high_quantile=args.high_quantile)

pre_epochs = 20
if pre_epochs > 0:
    pretrain(data_feature, adj, encoder, cell_features, cell_adj, expression_matrix, pre_epochs)

model = GCLink(encoder=encoder).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

best_auc = -1.0
best_path = os.path.join(model_dir, f"{args.net}_{args.cell_type}_best_model.pkl")

for epoch in range(args.epochs):
    model.train()
    running_loss = train(model, contrast_model_gene, contrast_model_cell, optimizer, scheduler, cell_features, cell_adj, expression_matrix)

    model.eval()
    _, _, _, _, _, _, _, score, _, _ = model(data_feature, adj, validation_data, cell_features, cell_adj, expression_matrix)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1], flag=args.flag)

    gamma_g_all, gamma_c_all = compute_epoch_gate_all(model, data_feature, adj, cell_features, cell_adj, expression_matrix)

    print(
        "Epoch:{} train loss:{:.5F} AUC:{:.3F} AUPR:{:.3F} gamma_g_all:{:.4f} gamma_c_all:{:.4f}".format(
            epoch + 1, running_loss, AUC, AUPR, gamma_g_all, gamma_c_all
        )
    )

    if AUC > best_auc:
        best_auc = AUC
        torch.save(model.state_dict(), best_path)

model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

_, _, _, _, _, _, _, score, _, _ = model(data_feature, adj, test_data, cell_features, cell_adj, expression_matrix)
if args.flag:
    score = torch.softmax(score, dim=1)
else:
    score = torch.sigmoid(score)

AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=args.flag)
print("test_AUC:{:.3F} test_AUPR:{:.3F}".format(AUC, AUPR))
