from torch_geometric.nn import GATConv, GraphConv, GCNConv, AGNNConv, EdgeConv
from torch_geometric.data import Data as gData
from torch_geometric.data import Batch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

roi_align = torchvision.ops.RoIAlign(output_size=(1, 1), spatial_scale=1.0, sampling_ratio=0)


def build_graph_batch(x, pre_valid, pre_bev, sample_range=10, edge_type='global'):
    """
    构建batch图
    :param x: 当前帧特征 (B, C, H, W)
    :param pre_valid: 上一帧掩码 (B, 1, H, W)，bool
    :param pre_bev: 上一帧特征 (B, C, H, W)
    :param roi_align: torchvision.ops.RoIAlign (只用于历史节点)
    :param sample_range: 采样范围 (当前帧取 (2*sample_range+1)^2 个点作为节点)
    :return:
        graph: torch_geometric.data.Batch
        center_inds: LongTensor, 当前帧节点在全局graph中的索引
    """
    B, C, H, W = x.shape
    device = x.device

    data_list = []
    cur_inds_range = []
    offset = 0  # 节点偏移量

    # 定义当前节点的相对偏移 (K,2)
    rel_coords = torch.stack(torch.meshgrid(
        torch.arange(-sample_range, sample_range + 1, device=device),
        torch.arange(-sample_range, sample_range + 1, device=device),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)  # (K,2)
    K = rel_coords.shape[0]

    for b in range(B):
        yi = x[b]  # (C, H, W)
        pre_valid_i = pre_valid[b, 0]  # (H, W)
        pre_bev_i = pre_bev[b]  # (C, H, W)

        # --- 历史节点 ---
        pre_inds = pre_valid_i.nonzero(as_tuple=False)  # (n,2) (y,x)
        n = len(pre_inds)
        if n == 0:
            continue

        boxes = torch.stack([
            pre_inds[:, 1] - sample_range,  # x1
            pre_inds[:, 0] - sample_range,  # y1
            pre_inds[:, 1] + sample_range,  # x2
            pre_inds[:, 0] + sample_range  # y2
        ], dim=-1).float()

        pre_nodes = roi_align(pre_bev_i.unsqueeze(0), [boxes])  # (n,C,1,1)
        pre_nodes = pre_nodes.view(n, C)

        # --- 当前节点 ---
        cur_nodes = yi.view(C, -1).T  # (H*W, C)

        # --- 节点拼接 ---
        graph_nodes = torch.cat([pre_nodes, cur_nodes], dim=0)  # (n+m,C)

        # --- 边构造 ---
        if edge_type == 'local':
            edge_src, edge_dst = [], []
            cur_start = n
            for i, (cy, cx) in enumerate(pre_inds.tolist()):
                y0, y1 = max(0, cy - sample_range), min(H, cy + sample_range + 1)
                x0, x1 = max(0, cx - sample_range), min(W, cx + sample_range + 1)
                # flatten index
                coords = (torch.arange(y0, y1, device=device).unsqueeze(1) * W +
                          torch.arange(x0, x1, device=device).unsqueeze(0))
                coords = coords.reshape(-1)
                dst = cur_start + coords
                src = torch.full((coords.numel(),), i, device=device, dtype=torch.long)
                edge_src.append(src)
                edge_dst.append(dst)
                edge_src.append(dst)
                edge_dst.append(src)
            edge_src = torch.cat(edge_src)
            edge_dst = torch.cat(edge_dst)
            edge_index = torch.stack([edge_src, edge_dst], dim=0)

        elif edge_type == 'local_fc':
            # 历史节点与当前所有历史节点对应区块全连接
            edge_src, edge_dst = [], []
            cur_ins = pre_inds.unsqueeze(1) + rel_coords.unsqueeze(0)  # (n,K,2)
            val_mask = (cur_ins[..., 0] >= 0) & (cur_ins[..., 0] < H) & \
                       (cur_ins[..., 1] >= 0) & (cur_ins[..., 1] < W)
            cur_ins = cur_ins[val_mask]  # (n*,2)
            cur_inds = cur_ins[:, 0] * W + cur_ins[:, 1]
            cur_start = n
            dst = cur_start + cur_inds  # (n*)
            dst = dst[None].repeat(n, 1).view(-1)  # (n*n*)
            src = torch.arange(n, device=device).repeat_interleave(cur_inds.size(0))  # (n*)
            edge_src.append(src)
            edge_dst.append(dst)
            edge_src.append(dst)
            edge_dst.append(src)
            edge_src = torch.cat(edge_src)
            edge_dst = torch.cat(edge_dst)
            edge_index = torch.stack([edge_src, edge_dst], dim=0)

        elif edge_type == 'global':
            # 历史节点全连到当前帧所有点
            his_inds = torch.arange(0, n, device=device)
            cur_inds = torch.arange(n, n + H * W, device=device)
            his_src = his_inds.repeat_interleave(cur_inds.size(0))
            his_dst = cur_inds.repeat(his_inds.size(0))
            cur_src, cur_dst = his_dst, his_src
            edge_src = torch.cat([his_src, cur_src], dim=0)
            edge_dst = torch.cat([his_dst, cur_dst], dim=0)
            edge_index = torch.stack([edge_src, edge_dst], dim=0)
        else:
            raise ValueError(f"edge_type {edge_type} not supported")

        # --- 保存图 ---
        data_list.append(gData(x=graph_nodes, edge_index=edge_index))

        # --- 当前帧节点索引(范围) ---
        cur_inds_range.append(torch.tensor([offset + n, offset + n + cur_nodes.size(0)], device=device))

        offset += graph_nodes.size(0)

    if len(data_list) == 0:
        return Batch(), torch.empty(0, dtype=torch.long, device=device)

    graph = Batch.from_data_list(data_list)
    # cur_inds_range = torch.cat(cur_inds_range, dim=0)  # (B,2)
    return graph, cur_inds_range


class GNN(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=64, num_layers=2, sample_range=10, cnn_type='GraphConv',
                 edge_type='global', dropout=0.1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.num_layers = num_layers
        self.cnn_type = cnn_type
        self.dropout = dropout
        self.edge_type = edge_type
        self.sample_range = sample_range

        # GNN 层
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 输入维度
        in_dim = in_ch
        for i in range(num_layers):
            out_dim = hidden_ch if i < num_layers - 1 else out_ch

            if cnn_type == 'EdgeConv':
                # EdgeConv 需要一个 MLP
                mlp = nn.Sequential(
                    nn.Linear(2 * in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                conv = EdgeConv(mlp)
            elif cnn_type == 'GATConv':
                # 默认1个head，可以改成 heads>1
                conv = GATConv(in_dim, out_dim, heads=1, concat=True, dropout=dropout)
            elif cnn_type == 'GraphConv':
                conv = GraphConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unsupported conv type: {cnn_type}")

            self.gnn_layers.append(conv)

            self.norms.append(nn.BatchNorm1d(out_dim))

            in_dim = out_dim

    def forward(self, x, pre_valid, pre_bev):
        B, C, H, W = x.shape
        graph, cur_inds_range = build_graph_batch(
            x, pre_valid, pre_bev, sample_range=self.sample_range, edge_type=self.edge_type
        )
        gnn_feat = graph.x
        for i, conv in enumerate(self.gnn_layers):
            gnn_out = conv(gnn_feat, graph.edge_index)

            gnn_out = self.norms[i](gnn_out)
            gnn_out = F.relu(gnn_out)
            gnn_out = F.dropout(gnn_out, p=self.dropout, training=self.training)
            if True:
                gnn_feat = gnn_out + gnn_feat  # 残差连接
            else:
                gnn_feat = gnn_out
        # 将 GNN 特征写回到 BEV 特征图
        bevs = []
        for inds in cur_inds_range:
            bev = gnn_feat[inds[0]:inds[1]].view(H, W, -1).permute(2, 0, 1)
            bevs.append(bev)
        x = torch.stack(bevs, dim=0)  # (B, C, H, W)
        return x

