import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors


class _HeteroGATHead(nn.Module):
    """
    单个 head 的异构 GAT：
    gene_x : [Ng, Fin_g]
    cell_x : [Nc, Fin_c]
    gene_adj      : [Ng, Ng]
    cell_adj      : [Nc, Nc]
    gene_cell_adj : [Ng, Nc] (gene->cell 0/1)
    """
    def __init__(self, gene_in_dim, cell_in_dim, out_dim, alpha=0.2, dropout=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

        # 节点类型特定线性变换
        self.gene_fc = nn.Linear(gene_in_dim, out_dim, bias=False)
        self.cell_fc = nn.Linear(cell_in_dim, out_dim, bias=False)

        # 4 种关系类型的注意力参数
        self.a_gg = nn.Parameter(torch.empty(2 * out_dim, 1))  # Gene <- Gene
        self.a_gc = nn.Parameter(torch.empty(2 * out_dim, 1))  # Gene <- Cell
        self.a_cc = nn.Parameter(torch.empty(2 * out_dim, 1))  # Cell <- Cell
        self.a_cg = nn.Parameter(torch.empty(2 * out_dim, 1))  # Cell <- Gene
        self.gate_g = nn.Sequential(
            nn.Linear(out_dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.gate_c = nn.Sequential(
            nn.Linear(out_dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gene_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.cell_fc.weight, gain=1.414)
        nn.init.xavier_uniform_(self.gate_g[0].weight, gain=1.414)
        nn.init.xavier_uniform_(self.gate_c[0].weight, gain=1.414)
        nn.init.constant_(self.gate_g[0].bias, -2.0)  # sigmoid(-2) ~ 0.12
        nn.init.constant_(self.gate_c[0].bias, -2.0)
        for p in [self.a_gg, self.a_gc, self.a_cc, self.a_cg]:
            nn.init.xavier_uniform_(p, gain=1.414)

    def forward(self, gene_x, cell_x, gene_adj, cell_adj, gene_cell_adj):
        Ng = gene_x.size(0)
        Nc = cell_x.size(0)
        d = self.out_dim

        # 1) 线性投影到公共空间
        gene_h = self.gene_fc(gene_x)   # [Ng, d]
        cell_h = self.cell_fc(cell_x)   # [Nc, d]

        # 2) 邻接矩阵转稠密（如果本来就是 dense，则 is_sparse=False）
        gene_adj_dense = gene_adj.to_dense() if gene_adj.is_sparse else gene_adj
        cell_adj_dense = cell_adj.to_dense() if cell_adj.is_sparse else cell_adj

        gc_adj = gene_cell_adj              # [Ng, Nc]
        cg_adj = gene_cell_adj.t()          # [Nc, Ng]

        # ---------- Gene <- Gene ----------
        Wh1g = gene_h @ self.a_gg[:d, :]    # [Ng, 1]
        Wh2g = gene_h @ self.a_gg[d:, :]    # [Ng, 1]
        e_gg = self.leakyrelu(Wh1g + Wh2g.T)  # [Ng, Ng]

        zero_gg = -9e15 * torch.ones_like(e_gg)
        att_gg = torch.where(gene_adj_dense > 0, e_gg, zero_gg)
        att_gg = F.softmax(att_gg, dim=1)
        if self.dropout > 0:
            att_gg = F.dropout(att_gg, p=self.dropout, training=self.training)
        gene_from_gene = att_gg @ gene_h     # [Ng, d]

        # ---------- Gene <- Cell ----------
        Wh1gc = gene_h @ self.a_gc[:d, :]
        Wh2gc = cell_h @ self.a_gc[d:, :]
        e_gc = self.leakyrelu(Wh1gc + Wh2gc.T)   # [Ng, Nc]

        zero_gc = -9e15 * torch.ones_like(e_gc)
        att_gc = torch.where(gc_adj > 0, e_gc, zero_gc)
        att_gc = F.softmax(att_gc, dim=1)
        if self.dropout > 0:
            att_gc = F.dropout(att_gc, p=self.dropout, training=self.training)
        gene_from_cell = att_gc @ cell_h         # [Ng, d]


        # ---------- Cell <- Cell ----------
        Wh1c = cell_h @ self.a_cc[:d, :]
        Wh2c = cell_h @ self.a_cc[d:, :]
        e_cc = self.leakyrelu(Wh1c + Wh2c.T)     # [Nc, Nc]

        zero_cc = -9e15 * torch.ones_like(e_cc)
        att_cc = torch.where(cell_adj_dense > 0, e_cc, zero_cc)
        att_cc = F.softmax(att_cc, dim=1)
        if self.dropout > 0:
            att_cc = F.dropout(att_cc, p=self.dropout, training=self.training)
        cell_from_cell = att_cc @ cell_h        # [Nc, d]

        # ---------- Cell <- Gene ----------
        Wh1cg = cell_h @ self.a_cg[:d, :]
        Wh2cg = gene_h @ self.a_cg[d:, :]
        e_cg = self.leakyrelu(Wh1cg + Wh2cg.T)  # [Nc, Ng]

        zero_cg = -9e15 * torch.ones_like(e_cg)
        att_cg = torch.where(cg_adj > 0, e_cg, zero_cg)
        att_cg = F.softmax(att_cg, dim=1)
        if self.dropout > 0:
            att_cg = F.dropout(att_cg, p=self.dropout, training=self.training)
        cell_from_gene = att_cg @ gene_h        # [Nc, d]

        gamma_g = self.gate_g(gene_h)  # [Ng, 1]
        gamma_c = self.gate_c(cell_h)  # [Nc, 1]

        # ===== cache for epoch-level statistics =====
        self._gamma_g_mean = gamma_g.mean().detach()
        self._gamma_c_mean = gamma_c.mean().detach()

        gene_out = self.leakyrelu(gene_h + gene_from_gene + gamma_g * gene_from_cell)
        cell_out = self.leakyrelu(cell_h + cell_from_cell + gamma_c * cell_from_gene)
        
        return gene_out, cell_out


class HeteroGATLayer(nn.Module):
    """
    多头异构 GAT：
    - 每个 head 是一个 _HeteroGATHead
    - 输出对各 head 取平均，因此输出维度仍为 out_dim
    """
    def __init__(self, gene_in_dim, cell_in_dim, out_dim,
                 num_heads=1, alpha=0.2, dropout=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            _HeteroGATHead(
                gene_in_dim=gene_in_dim,
                cell_in_dim=cell_in_dim,
                out_dim=out_dim,
                alpha=alpha,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])

    def forward(self, gene_x, cell_x, gene_adj, cell_adj, gene_cell_adj):
        gene_outs = []
        cell_outs = []
        for head in self.heads:
            g, c = head(gene_x, cell_x, gene_adj, cell_adj, gene_cell_adj)
            gene_outs.append(g)
            cell_outs.append(c)

        # [num_heads, Ng, d] -> [Ng, d]
        gene_out = torch.mean(torch.stack(gene_outs, dim=0), dim=0)
        # [num_heads, Nc, d] -> [Nc, d]
        cell_out = torch.mean(torch.stack(cell_outs, dim=0), dim=0)

        return gene_out, cell_out



class GENELink(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim, num_head1, num_head2,
                 alpha, device, type, reduction,
                 cell_feature_dim=None, cell_hidden_dim=64, top_k=20,
                 dynamic_gc: bool = True, dynamic_gc_detach: bool = True):
        print("开始 GENELink（异构GAT版）初始化")
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction
        self.top_k = top_k
        self.dynamic_gc = dynamic_gc
        self.dynamic_gc_detach = dynamic_gc_detach

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            # 仍保留原来的多头 GAT 参数（作为 fallback）
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim
        else:
            raise TypeError("reduction must be 'mean' or 'concate'")

        # ====== 原基因 GAT（仅在没有细胞特征时 fallback 使用） ======
        self.ConvLayer1 = nn.ModuleList([
            AttentionLayer(input_dim, hidden1_dim, alpha) for _ in range(num_head1)
        ])
        self.ConvLayer2 = nn.ModuleList([
            AttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)
        ])

        # ====== TF / Target 全连接层（保持不变） ======
        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim, hidden3_dim)

        self.tf_linear2 = nn.Linear(hidden3_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)

        # ====== 异构 GAT 相关 ======
        self.use_cell_features = cell_feature_dim is not None

        if self.use_cell_features:
            # 两层异构 GAT：layer1: (input_dim, cell_feature_dim) -> hidden1_dim
            #               layer2: (hidden1_dim, hidden1_dim)      -> hidden2_dim
            self.hetero_layer1 = HeteroGATLayer(
                gene_in_dim=input_dim,
                cell_in_dim=cell_feature_dim,
                out_dim=hidden1_dim,   # 单头输出维度
                num_heads=num_head1,
                alpha=alpha
            )
            self.hetero_layer2 = HeteroGATLayer(
                gene_in_dim=hidden1_dim,
                cell_in_dim=hidden1_dim,
                out_dim=hidden2_dim,
                num_heads=num_head2,
                alpha=alpha
            )

            # ===== gene-cell 注意力参数（用于 get_top_cell_features，当前你 forward 不走它也没问题）=====
            self.gc_q_proj_tf = nn.Linear(output_dim, output_dim, bias=False)        # Q: tf_embed
            self.gc_q_proj_target = nn.Linear(output_dim, output_dim, bias=False)    # Q: target_embed
            self.gc_k_proj = nn.Linear(output_dim, output_dim, bias=False)           # K: cell_embed
            self.gc_att_vec_tf = nn.Parameter(torch.empty(2 * output_dim, 1))
            self.gc_att_vec_target = nn.Parameter(torch.empty(2 * output_dim, 1))
            self.gc_leakyrelu = nn.LeakyReLU(alpha)

            # 将细胞 embedding 从 hidden2_dim 投影到 output_dim，方便后续解码
            self.cell_proj = nn.Linear(hidden2_dim, output_dim)

            if self.type == 'MLP':
                # 解码器输入：tf/target + 各自 cell 特征 => 4*output_dim
                self.linear = nn.Linear(4 * output_dim, 1)
        else:
            if self.type == 'MLP':
                # 不用细胞特征时：tf/target => 2*output_dim
                self.linear = nn.Linear(2 * output_dim, 1)

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()
        print("GENELink 初始化完成（已启用异构 GAT 编码）")

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

        if self.use_cell_features:
            nn.init.xavier_uniform_(self.cell_proj.weight, gain=1.414)
            nn.init.xavier_uniform_(self.gc_q_proj_tf.weight, gain=1.414)
            nn.init.xavier_uniform_(self.gc_q_proj_target.weight, gain=1.414)
            nn.init.xavier_uniform_(self.gc_k_proj.weight, gain=1.414)
            nn.init.xavier_uniform_(self.gc_att_vec_tf, gain=1.414)
            nn.init.xavier_uniform_(self.gc_att_vec_target, gain=1.414)

        with torch.no_grad():
            self.logit_scale.fill_(1.0)

    # --------- gene-cell 邻接（静态 bootstrap：表达 top-k）---------
    def build_gene_cell_adj(self, expression_matrix, top_k=None):
        """
        expression_matrix: [n_genes, n_cells] torch.Tensor or numpy array
        return: [n_genes, n_cells] float tensor (0/1)
        """
        if top_k is None:
            top_k = self.top_k

        if not torch.is_tensor(expression_matrix):
            expression_matrix = torch.tensor(expression_matrix, dtype=torch.float32)

        expression_matrix = expression_matrix.to(self.device)
        n_genes, n_cells = expression_matrix.shape
        k = min(top_k, n_cells)

        _, top_idx = torch.topk(expression_matrix, k=k, dim=1)  # [n_genes, k]

        adj = torch.zeros((n_genes, n_cells), device=expression_matrix.device, dtype=torch.float32)
        row_idx = torch.arange(n_genes, device=expression_matrix.device).unsqueeze(1)
        adj[row_idx, top_idx] = 1.0
        return adj

    # --------- gene-cell 邻接（动态：embedding 相似度 top-k）---------
    def build_dynamic_gene_cell_adj(self, gene_embed, cell_embed, top_k=None, detach: bool = False):
        """
        gene_embed : [Ng, d]
        cell_embed : [Nc, d]
        return     : [Ng, Nc] (0/1)
        """
        if detach:
            gene_embed = gene_embed.detach()
            cell_embed = cell_embed.detach()

        if top_k is None:
            top_k = self.top_k

        gene_embed = F.normalize(gene_embed, dim=1)
        cell_embed = F.normalize(cell_embed, dim=1)
        sim = torch.matmul(gene_embed, cell_embed.T)  # [Ng, Nc]

        k = min(top_k, sim.size(1))
        _, top_idx = torch.topk(sim, k=k, dim=1)

        gene_cell_adj = torch.zeros_like(sim)
        row_idx = torch.arange(sim.size(0), device=sim.device).unsqueeze(1)
        gene_cell_adj[row_idx, top_idx] = 1.0
        return gene_cell_adj

    # ---------- 编码：异构 GAT + fallback ----------
    def encode(self, x, adj, cell_features=None, cell_adj=None, expression_matrix=None):
        """
        x                : 基因特征 [N_genes, ?]（你的项目里通常是 [N_genes, N_cells]）
        adj              : 基因邻接 (稀疏 COO tensor)
        cell_features    : 细胞特征 [N_cells, ?]
        cell_adj         : 细胞邻接 [N_cells, N_cells]
        expression_matrix: 原始表达矩阵 [N_genes, N_cells]
        return:
            gene_embed : [N_genes, hidden2_dim]
            cell_embed : [N_cells, output_dim] or None
        """
        if self.use_cell_features and (cell_features is not None) and (cell_adj is not None) and (expression_matrix is not None):
            # 1) bootstrap：表达 top-k
            gene_cell_adj_init = self.build_gene_cell_adj(expression_matrix)

            # 2) 第 1 层异构 GAT
            gene_h1, cell_h1 = self.hetero_layer1(
                x, cell_features, adj, cell_adj, gene_cell_adj_init
            )

            # 3) 动态重连：embedding top-k（用于第 2 层）
            if self.dynamic_gc:
                gene_cell_adj_used = self.build_dynamic_gene_cell_adj(
                    gene_h1, cell_h1,
                    top_k=self.top_k,
                    detach=self.dynamic_gc_detach
                )
            else:
                gene_cell_adj_used = gene_cell_adj_init

            # 4) 第 2 层异构 GAT
            gene_h2, cell_h2 = self.hetero_layer2(
                gene_h1, cell_h1, adj, cell_adj, gene_cell_adj_used
            )

            gene_embed = gene_h2
            cell_embed = self.cell_proj(cell_h2)  # [N_cells, output_dim]
            return gene_embed, cell_embed

        # ---------- fallback：只在基因图上做原 AttentionLayer ----------
        if self.reduction == 'concate':
            x1 = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x1 = F.elu(x1)
        elif self.reduction == 'mean':
            x1 = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x1 = F.elu(x1)
        else:
            raise TypeError

        out = torch.mean(torch.stack([att(x1, adj) for att in self.ConvLayer2]), dim=0)
        return out, None

    # ---------- 基因对应的细胞上下文特征（原逻辑：表达 top-k + 注意力加权） ----------
    def get_top_cell_features(self, expression_matrix, cell_features, use_target: bool = False):
        n_genes, n_cells = expression_matrix.shape
        top_k = min(self.top_k, n_cells)

        _, top_indices = torch.topk(expression_matrix, k=top_k, dim=1)  # [n_genes, top_k]

        if not use_target:
            Q_embed = self.tf_ouput
            q_proj = self.gc_q_proj_tf
            att_vec = self.gc_att_vec_tf
        else:
            Q_embed = self.target_output
            q_proj = self.gc_q_proj_target
            att_vec = self.gc_att_vec_target

        Q = q_proj(Q_embed)  # [n_genes, d]
        K_all = self.gc_k_proj(cell_features)  # [n_cells, d]
        V_all = cell_features  # [n_cells, d]

        K_top = K_all[top_indices]  # [n_genes, top_k, d]
        V_top = V_all[top_indices]  # [n_genes, top_k, d]

        Q_expanded = Q.unsqueeze(1).expand(-1, top_k, -1)           # [n_genes, top_k, d]
        concat_qk = torch.cat([Q_expanded, K_top], dim=-1)          # [n_genes, top_k, 2d]

        e = torch.matmul(concat_qk, att_vec).squeeze(-1)            # [n_genes, top_k]
        e = self.gc_leakyrelu(e)

        alpha = F.softmax(e, dim=1)                                 # [n_genes, top_k]
        result = torch.sum(V_top * alpha.unsqueeze(-1), dim=1)      # [n_genes, d]
        return result

    # ---------- 动态 gene->cell 上下文（embedding top-k + mean pooling） ----------
    def get_dynamic_gene_cell_features(self, gene_embed, cell_embed):
        gene_embed = F.normalize(gene_embed, dim=1)
        cell_embed = F.normalize(cell_embed, dim=1)

        sim = torch.matmul(gene_embed, cell_embed.T)  # [Ng, Nc]
        k = min(self.top_k, sim.size(1))
        _, top_idx = torch.topk(sim, k=k, dim=1)

        cell_top = cell_embed[top_idx]   # [Ng, k, d]
        return cell_top.mean(dim=1)      # [Ng, d]

    # ---------- 解码 ----------
    def decode(self, tf_embed, target_embed, tf_cell_features=None, target_cell_features=None):
        if self.type == 'dot':
            if self.use_cell_features and tf_cell_features is not None and target_cell_features is not None:
                tf_combined = torch.cat([tf_embed, tf_cell_features], dim=1)
                target_combined = torch.cat([target_embed, target_cell_features], dim=1)
            else:
                tf_combined = tf_embed
                target_combined = target_embed

            tf_norm = F.normalize(tf_combined, p=2, dim=1)
            target_norm = F.normalize(target_combined, p=2, dim=1)

            dot = torch.sum(tf_norm * target_norm, dim=1, keepdim=True)  # [-1,1]
            logits = dot * self.logit_scale
            return logits

        elif self.type == 'cosine':
            if self.use_cell_features and tf_cell_features is not None and target_cell_features is not None:
                tf_combined = torch.cat([tf_embed, tf_cell_features], dim=1)
                target_combined = torch.cat([target_embed, target_cell_features], dim=1)
                prob = torch.cosine_similarity(tf_combined, target_combined, dim=1).view(-1, 1)
            else:
                prob = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1, 1)
            return prob

        elif self.type == 'MLP':
            if self.use_cell_features and tf_cell_features is not None and target_cell_features is not None:
                h = torch.cat([tf_embed, target_embed, tf_cell_features, target_cell_features], dim=1)
            else:
                h = torch.cat([tf_embed, target_embed], dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))

    # ---------- 前向传播 ----------
    def forward(self, x, adj, train_sample, cell_features=None, cell_adj=None, expression_matrix=None):
        embed, cell_embed = self.encode(x, adj, cell_features, cell_adj, expression_matrix)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)

        # 保存下来，供 get_top_cell_features 使用（即使你当前 forward 走 dynamic，也不影响）
        self.tf_ouput = tf_embed
        self.target_output = target_embed

        if self.use_cell_features and (cell_embed is not None) and (expression_matrix is not None):
            tf_gene_cell_features = self.get_dynamic_gene_cell_features(tf_embed, cell_embed)
            target_gene_cell_features = self.get_dynamic_gene_cell_features(target_embed, cell_embed)

            tf_cell_features = tf_gene_cell_features[train_sample[:, 0]]
            target_cell_features = target_gene_cell_features[train_sample[:, 1]]
        else:
            tf_cell_features = None
            target_cell_features = None

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target, tf_cell_features, target_cell_features)
        return embed, tf_embed, target_embed, pred, cell_embed

    def get_embedding(self):
        return self.tf_ouput, self.target_output




class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T, negative_slope=self.alpha)
        return e

    def forward(self, x, adj):

        h = torch.matmul(x, self.weight)  # h = XW
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass

        output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


class CellGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.2):
        super(CellGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight1 = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.weight2 = nn.Parameter(torch.FloatTensor(hidden_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight2.data, gain=1.414)

    def forward(self, x, adj):
        # x: 细胞特征矩阵 [N_cells, input_dim]
        # adj: 细胞邻接矩阵 [N_cells, N_cells]
        # 第一层GCN
        h = torch.matmul(x, self.weight1)
        h = torch.matmul(adj, h)
        h = F.leaky_relu(h, negative_slope=self.alpha)

        # 第二层GCN
        h = torch.matmul(h, self.weight2)
        h = torch.matmul(adj, h)
        h = F.leaky_relu(h, negative_slope=self.alpha)

        return h


def create_cell_adjacency_matrix(cell_features, n_neighbors=20):
    """
    基于细胞特征的KNN创建邻接矩阵
    cell_features: 细胞特征矩阵 [N_cells, feature_dim]
    return: 邻接矩阵 [N_cells, N_cells]
    """
    n_cells = cell_features.shape[0]

    # 使用KNN找到每个细胞的邻居
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 包括自身
    knn.fit(cell_features)
    distances, indices = knn.kneighbors(cell_features)

    # 创建邻接矩阵
    adj = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        adj[i, indices[i]] = 1.0

    # 确保对称性（可选，取决于你的需求）
    adj = np.maximum(adj, adj.T)

    # 归一化
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    adj_normalized = adj * r_inv[:, np.newaxis]
    result = torch.FloatTensor(adj_normalized)
    return torch.FloatTensor(adj_normalized)