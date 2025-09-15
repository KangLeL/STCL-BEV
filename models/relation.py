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



class GCD(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(GCD, self).__init__()
        # GCM组件
        self.gcm = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),  # W_k投影
            nn.Softmax(dim=2)  # 空间Softmax
        )

        # TFF变换层（检测分支）
        self.det_bottleneck = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LayerNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        # TFF变换层（ReID分支）
        self.reid_bottleneck = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LayerNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

    def forward(self, x):
        """ 输入: x [B, C, H, W]
            输出:
                d - 检测特征 [B, C, H, W]
                r - ReID特征 [B, C, H, W]
        """
        B, C, H, W = x.shape

        # =====================
        # 1. GCM阶段（公式1）
        # =====================
        attn = self.gcm(x)  # [B, 1, H, W]
        attn = attn.view(B, 1, -1)  # [B, 1, H*W]
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        z = torch.bmm(x_flat, attn.transpose(1, 2))  # [B, C, 1]
        z = z.view(B, C, 1, 1)  # 全局上下文向量 [B, C, 1, 1]

        # =====================
        # 2. TFF阶段（公式2-3）
        # =====================
        z_flat = z.view(B, C)  # 展平为向量[B, C]

        # 检测分支变换
        d_transform = self.det_bottleneck(z_flat)  # [B, C]
        d_transform = d_transform.view(B, C, 1, 1).expand_as(x)

        # ReID分支变换
        r_transform = self.reid_bottleneck(z_flat)  # [B, C]
        r_transform = r_transform.view(B, C, 1, 1).expand_as(x)

        # 残差连接（恢复低层特征）
        d = x + d_transform  # 检测特征
        r = x + r_transform  # ReID特征

        return d, r