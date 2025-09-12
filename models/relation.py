import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRelationHead(nn.Module):
    def __init__(self, feat_dim=256, hidden=128):
        super().__init__()
        self.q = nn.Linear(feat_dim, hidden)
        self.k = nn.Linear(feat_dim, hidden)
        self.v = nn.Linear(feat_dim, feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, roi_feats, boxes=None):
        # roi_feats: (B, N, C)
        B, N, C = roi_feats.shape
        Q = self.q(roi_feats)  # (B, N, H)
        K = self.k(roi_feats)
        V = self.v(roi_feats)

        # 计算 batch 内的 attention: (B, N, N)
        sim = torch.matmul(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(sim, dim=-1)
        agg = torch.matmul(attn, V)  # (B, N, C)

        out = F.relu(self.out(agg + roi_feats))  # (B, N, C)
        return out
