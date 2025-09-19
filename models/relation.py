import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.modules.ms_deform_attn import MSDeformAttn
class DeformableMHAttention(nn.Module):
    """
    Deformable Multi-Head Attention (DMHA).
    基于 feature map 的稀疏采样版本。
    """
    def __init__(self, d_model, nhead=4, num_keys=9):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.num_keys = num_keys
        self.head_dim = d_model // nhead

        # Q, K, V projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # offsets + attention weights per head
        self.offset_mlp = nn.Linear(d_model, 2 * nhead * num_keys)
        self.attn_mlp   = nn.Linear(d_model, nhead * num_keys)

        # output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, H, W):
        """
        x: (B, S, C) where S=H*W
        returns: (B, S, C)
        """
        B, S, C = x.shape
        device = x.device

        Q = self.q_proj(x)  # (B,S,C)
        V = self.v_proj(x).view(B, S, self.nhead, self.head_dim)

        # predict offsets and attention weights
        offsets = self.offset_mlp(Q)
        attn_logits = self.attn_mlp(Q)

        offsets = offsets.view(B, S, self.nhead, self.num_keys, 2)
        attn = attn_logits.view(B, S, self.nhead, self.num_keys)
        attn = F.softmax(attn, dim=-1)

        # base grid coords (normalized [-1,1])
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        base_grid = torch.stack((xx, yy), dim=-1).view(1, S, 1, 1, 2)  # (1,S,1,1,2)

        # normalize offsets
        offsets[..., 0] = offsets[..., 0] / ((W - 1) / 2.0)
        offsets[..., 1] = offsets[..., 1] / ((H - 1) / 2.0)

        # sampling grid = base + offset
        sampling_grids = base_grid + offsets  # (B,S,nhead,num_keys,2)

        # reshape to feed grid_sample
        feat_map = V.permute(0, 2, 1, 3).contiguous().view(B*self.nhead, H, W, self.head_dim)
        feat_map = feat_map.permute(0, 3, 1, 2)  # (B*nhead, dim, H, W)

        grids = sampling_grids.view(B*self.nhead, S*self.num_keys, 2)
        grids = grids.view(B*self.nhead, H, W, self.num_keys, 2)  # reshape per location

        # 为简化：直接用 scatter/gather 实现，或者近似用全局 attn (工程化时换更高效的 CUDA op)
        # 这里我直接近似实现：取 base+offset 对应的索引最近点（可优化）
        grids = grids.clamp(-1, 1)
        # 直接用 bilinear 插值
        sampled = F.grid_sample(
            feat_map,
            sampling_grids.view(B*self.nhead, S*self.num_keys, 1, 2),
            align_corners=True
        )  # (B*nhead, dim, S*num_keys, 1)

        sampled = sampled.view(B, self.nhead, self.head_dim, S, self.num_keys)
        sampled = sampled.permute(0, 3, 1, 4, 2)  # (B,S,nhead,num_keys,dim)

        # 加权求和
        attn = attn.unsqueeze(-1)  # (B,S,nhead,num_keys,1)
        out = (sampled * attn).sum(dim=3)  # (B,S,nhead,dim)

        out = out.reshape(B, S, C)
        out = self.out_proj(out)
        return out


class DeformableTransformerBlock(nn.Module):
    def __init__(self, d_model, Y, X, dim_ff=256, dropout=0.1, n_points=4):
        super().__init__()
        self.attn = MSDeformAttn(d_model,n_levels=4, n_heads=8, n_points=n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, Y - 0.5, Y, dtype=torch.float32),
                                      torch.linspace(0.5, X - 0.5, X, dtype=torch.float32))
        self.ref = torch.stack((ref_x, ref_y), -1).reshape([-1, 2])  # (Y*X,2)
        self.ref[:, 0] /= X
        self.ref[:, 1] /= Y
        self.ref = self.ref[:, None, None,:].expand(-1, -1, n_points, -1)
        self.input_spatial_shapes = torch.tensor([[Y, X]], dtype=torch.long)
        self.input_level_start_index = torch.tensor([0], dtype=torch.long)


    def forward(self, x):
        # deformable attention
        B, S, C = x.shape
        # N, Len_q, n_levels, n_points, 2
        reference_points = self.ref.unsqueeze(0).expand(B, -1, -1, -1, -1)
        x2 = self.attn(x, reference_points.to(x.device), x,
                       self.input_spatial_shapes.to(x.device), self.input_level_start_index.to(x.device))
        x = x + x2
        x = self.norm1(x)
        # FFN
        x2 = self.ff(x)
        x = x + x2
        x = self.norm2(x)
        return x


class GTE(nn.Module):
    """
    GTE with deformable attention replacing Transformer attention.
    """
    def __init__(self, in_ch, Y, X, num_layers=1, dim_ff=256):
        super().__init__()
        self.in_ch = in_ch
        self.blocks = nn.ModuleList([
            DeformableTransformerBlock(in_ch, Y, X, dim_ff=dim_ff)
            for _ in range(num_layers)
        ])
        self.cr = ChannelRelationBlock(in_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)  # (B,S,C)
        for blk in self.blocks:
            seq = blk(seq)
        feat = seq.permute(0, 2, 1).view(B, C, H, W)
        return self.cr(feat)


class ChannelRelationBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.conv(self.gpool(x))
        y = self.relu(y).view(B, -1)
        y = self.fc(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x + x * y


class GCD(nn.Module):
    def __init__(self, in_channels, reduction=8):
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