import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .ops.modules import MSDeformAttn
import copy


class DeformTransWorldFeat(nn.Module):
    def __init__(self, num_cam, Rworld_shape, base_dim, hidden_dim=128, dropout=0.1, nhead=8, dim_feedforward=512,
                 n_points=4, stride=2, use_Temporal=False, use_new_deformable=False):
        super(DeformTransWorldFeat, self).__init__()
        self.stride = stride
        self.downsample = nn.Sequential(nn.Conv2d(base_dim, hidden_dim, 3, stride, 1), nn.ReLU(), )

        encoder_layer_list = nn.ModuleList()
        reference_points_sample = create_reference_map(Rworld_shape, n_points, n_cam=num_cam, downsample=stride)
        if use_Temporal:
            encoder_layer_list.append(DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward, dropout,
                                                                        n_levels=1, n_heads=nhead, n_points=n_points))
        encoder_layer_list.append(DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward, dropout,
                                                                    n_levels=num_cam, n_heads=nhead,
                                                                    n_points=n_points))
        reference_points = reference_points_sample.repeat(num_cam, 1, 1, 1)  # (H*W), n_cam, n_points, 2
        self.encoder = DeformableTransformerEncoder(encoder_layer_list, 3, reference_points,
                                                    use_new_deformable=use_new_deformable)
        self.pos_embedding = create_pos_embedding(np.array(Rworld_shape) // stride, hidden_dim // 2)
        self.lvl_embedding = nn.Parameter(torch.Tensor(num_cam, hidden_dim))

        self.merge_linear = nn.Sequential(nn.Conv2d(hidden_dim * num_cam, hidden_dim, 1), nn.ReLU())
        self.upsample = nn.Sequential(nn.Upsample(Rworld_shape, mode='bilinear', align_corners=False),
                                      nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1), nn.ReLU(), )
        self._reset_parameters()

    def forward(self, x, visualize=False, pre_world_features=None):
        B, N, C, H, W = x.shape

        x = self.downsample(x.view(B * N, C, H, W))

        _, C, H, W = x.shape

        if pre_world_features is None:
            # pre_world_features = x.view(B, N, C, H, W).permute(0, 2, 3, 4, 1).mean(-1).contiguous().view(B, C, -1) \
            #     .permute(0, 2, 1)
            pre_world_features = torch.zeros([B, H * W, C], device=x.device)

        else:
            pre_world_features = F.interpolate(pre_world_features, size=(H, W), mode='bilinear',
                                               align_corners=False)

        src_flatten = x.view(B, N, C, H, W).permute(0, 1, 3, 4, 2).contiguous().view([B, N * H * W, C])
        lvl_pos_embed_flatten = (self.pos_embedding.to(x.device).flatten(2).transpose(1, 2).unsqueeze(1) +
                                 self.lvl_embedding.view([B, N, 1, C])).view([B, N * H * W, C])
        spatial_shapes = torch.as_tensor(np.array([[H, W]] * N), dtype=torch.long, device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones([B, N, 2], device=x.device)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              pre_world_features=pre_world_features)
        merged_feat = self.merge_linear(memory.view(B, N, H, W, C).permute(0, 1, 4, 2, 3).contiguous().
                                        view(B, N * C, H, W))
        merged_feat = self.upsample(merged_feat)
        return merged_feat

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.lvl_embedding)


def create_reference_map(Rworld_shape, n_points=4, downsample=2, visualize=False, n_cam=7):
    H, W = Rworld_shape  # H,W; N_row,N_col
    H, W = H // downsample, W // downsample
    ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float32),
                                  torch.linspace(0.5, W - 0.5, W, dtype=torch.float32))
    ref = torch.stack((ref_x, ref_y), -1).reshape([-1, 2])
    ref = ref.unsqueeze(1).unsqueeze(2).repeat([1, n_cam, n_points, 1])  # H*W, n_cam, n_points, 2

    ref = ref / torch.tensor([W, H], dtype=torch.float32)  # 归一化到0-1之间
    return ref


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, value, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, value, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_list, num_layers, reference_points=None, use_new_deformable=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer_list, num_layers)
        self.num_layers = num_layers
        self.reference_points = reference_points
        self.use_new_deformable = use_new_deformable
        self.use_Temporal = False
        if len(encoder_layer_list) > 1:
            self.use_Temporal = True
            self.TSA_ref = reference_points[:, 0:1, :, :]  # (H*W), 1, n_points, 2

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                pre_world_features=None):
        output = src
        if self.reference_points is None:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        else:
            reference_points = self.reference_points.unsqueeze(0).repeat([src.shape[0], 1, 1, 1, 1]).to(src.device)
            if self.use_Temporal:
                TSA_ref = self.TSA_ref.unsqueeze(0).repeat([src.shape[0], 1, 1, 1, 1]).to(src.device)

        for i, layer in enumerate(self.layers):
            input_tensor = src if self.use_new_deformable else output
            if self.use_Temporal:
                if i % 2 == 0:
                    output = layer(output, pre_world_features, pos, TSA_ref, spatial_shapes[0:1, :],
                                   level_start_index[0:1],
                                   padding_mask)
                else:
                    output = layer(output, input_tensor, pos, reference_points, spatial_shapes, level_start_index,
                                   padding_mask)
            else:
                output = layer(output, input_tensor, pos, reference_points, spatial_shapes, level_start_index,
                               padding_mask)

        return output


def _get_clones(module, N):
    M_list = nn.ModuleList()
    if isinstance(module, nn.ModuleList):
        for _ in range(N):
            for m in module:
                M_list.append(copy.deepcopy(m))
        return M_list
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def create_pos_embedding(img_size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    H, W = img_size
    not_mask = torch.ones([1, H, W])
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos

# class TSAEncoderLayer(nn.Module):
#     def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, reference_points=None):
#         super().__init__()
#         self.reference_points = reference_points
#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#
#         # ffn
#         self.linear1 = nn.Linear(d_model, d_model * 4)
#         self.dropout2 = nn.Dropout(0.1)
#         self.linear2 = nn.Linear(d_model * 4, d_model)
#         self.dropout3 = nn.Dropout(0.1)
#         self.norm2 = nn.LayerNorm(d_model)
