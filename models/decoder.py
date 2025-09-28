import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.encoder import freeze_bn, UpsamplingConcat

from .ops.modules import MSDeformAttn

from .relation import GTE, GCD

from .gnn import GNN, GNN_GCD



class Decoder(nn.Module):
    def __init__(self, Y, X, in_channels, n_classes, n_ids, useGCD=False, fusion_type=None, use_GTE=False, gnn_type=None,
                 gnn_layers=0):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(backbone)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.reid_feat = 64
        self.feat2d = 128

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingConcat(256 + 128, 256)
        self.up2_skip = UpsamplingConcat(256 + 64, 256)
        self.up1_skip = UpsamplingConcat(256 + in_channels, shared_out_channels)



        # bev
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
        )
        self.instance_center_head[-1].bias.data.fill_(-2.19)

        self.instance_size_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 3, kernel_size=1, padding=0),
        )
        self.instance_rot_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 8, kernel_size=1, padding=0),
        )

        # img
        self.img_center_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, n_classes, kernel_size=1, padding=0),
        )
        self.img_offset_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )
        self.img_size_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, 2, kernel_size=1, padding=0),
        )

        # re_id
        self.id_feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.reid_feat, kernel_size=1, padding=0),
        )
        self.img_id_feat_head = nn.Sequential(
            nn.Conv2d(self.feat2d, self.feat2d, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(self.feat2d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat2d, self.reid_feat, kernel_size=1, padding=0),
        )
        self.emb_scale = math.sqrt(2) * math.log(n_ids - 1)

        self.history_bev = None

        self.fusion_type = fusion_type
        self.spatial_context_flag = self.fusion_type is not None
        self.relationFlag = False
        self.GCDFlag = useGCD
        self.use_GTE = use_GTE
        self.gnn_type = gnn_type
        self.gnn_layers = gnn_layers
        assert self.fusion_type in [None, 'concat', 'deformAttn']
        if self.fusion_type == 'concat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(2 * shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True)
            )
        elif self.fusion_type == 'deformAttn':
            self.self_attn = MSDeformAttn(shared_out_channels, 1, 8, 4)

        if self.GCDFlag:
            # self.gcd = GCD(shared_out_channels, 8)
            self.gcd = GNN_GCD(shared_out_channels, shared_out_channels)
        if self.use_GTE:
            self.GTE = GTE(shared_out_channels,Y,X)

        if self.gnn_type is not None and self.gnn_layers > 0:
            self.norm = nn.BatchNorm2d(shared_out_channels)
            self.gnn = GNN(shared_out_channels, shared_out_channels, hidden_ch=shared_out_channels, num_layers=self.gnn_layers,
                           cnn_type=self.gnn_type, edge_type='local', dropout=0.1)


    def unet(self, x):
        b, c, h, w = x.shape

        # pad input
        m = 8
        ph, pw = math.ceil(h / m) * m - h, math.ceil(w / m) * m - w
        x = torch.nn.functional.pad(x, (ph, pw))

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        # Unpad
        x = x[..., ph // 2:h + ph // 2, pw // 2:w + pw // 2]
        return x

    def spatial_context(self, bev1, bev2):
        if bev2 is None:
            return bev1
        if self.fusion_type == 'concat':
            # 沿通道维拼接
            bev_cat = torch.cat([bev1, bev2], dim=1)  # [B, 2C, H, W]
            fused = self.fusion_conv(bev_cat)  # [B, out_channels, H, W]
            return fused

    def reset(self):
        self.history_bev = None

    def forward(self, x, feat_cams, history_bev=None, pre_valid_bev=None):
        b, c, h, w = x.shape
        if history_bev and self.spatial_context_flag:
            self.history_bev = None
            with torch.no_grad():
                for bev in history_bev:
                    bev = self.unet(bev)
                    if self.history_bev is None:
                        self.history_bev = bev
                    else:
                        self.history_bev = self.spatial_context(bev, self.history_bev)


        # x = self.unet(x)  # B, C, H, W
        if self.gnn_layers > 0 and self.gnn_type is not None:
            x = self.norm(x)
            x = F.relu(x)
            if self.history_bev is None:
                self.history_bev = x.detach()
            else:
                x = self.gnn(x, pre_valid_bev, self.history_bev)
                self.history_bev = x.detach()

        if self.spatial_context_flag:
            x = self.spatial_context(x, self.history_bev)
            self.history_bev = x.detach()


        # Extra upsample to (2xH, 2xW)
        # x = self.up_sample_2x(x)

        if self.GCDFlag:
            # x_det, x_id = self.gcd(x)
            # if self.use_GTE:
            #     x_id = self.GTE(x_id)
            # bev
            instance_center_output = self.instance_center_head(x)
            instance_offset_output = self.instance_offset_head(x)
            instance_size_output = self.instance_size_head(x)
            instance_rot_output = self.instance_rot_head(x)

            x_id = self.gcd(x, instance_center_output)

            instance_id_feat_output = self.emb_scale * F.normalize(self.id_feat_head(x_id), dim=1)
        else:

            # bev
            instance_center_output = self.instance_center_head(x)
            instance_offset_output = self.instance_offset_head(x)
            instance_size_output = self.instance_size_head(x)
            instance_rot_output = self.instance_rot_head(x)
            instance_id_feat_output = self.emb_scale * F.normalize(self.id_feat_head(x), dim=1)

        if self.relationFlag:
            # 1. 提取 top-K 目标
            topk_feats, topk_inds = extract_topk_features(
                instance_id_feat_output, instance_center_output, K=30
            )  # [B, K, C]

            # 2. 做关系建模
            B, K, C = topk_feats.shape
            feats_rel = self.pairwise_relation_head(topk_feats)  # [B, K, C]

            # 3. scatter 回去
            id_feat_flat = instance_id_feat_output.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
            for bc in range(B):
                id_feat_flat[bc, topk_inds[bc]] = feats_rel[bc]
            instance_id_feat_output = id_feat_flat.permute(0, 2, 1).view_as(instance_id_feat_output)

        # img
        img_center_output = self.img_center_head(feat_cams)  # B*S,1,H/8,W/8
        img_offset_output = self.img_offset_head(feat_cams)  # B*S,2,H/8,W/8
        img_size_output = self.img_size_head(feat_cams)  # B*S,2,H/8,W/8
        img_id_feat_output = self.emb_scale * F.normalize(self.img_id_feat_head(feat_cams), dim=1)  # B*S,C,H/8,W/8

        return {
            # bev
            'raw_feat': x,
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_size': instance_size_output.view(b, *instance_size_output.shape[1:]),
            'instance_rot': instance_rot_output.view(b, *instance_rot_output.shape[1:]),
            'instance_id_feat': instance_id_feat_output.view(b, *instance_id_feat_output.shape[1:]),
            # img
            'img_center': img_center_output,
            'img_offset': img_offset_output,
            'img_size': img_size_output,
            'img_id_feat': img_id_feat_output,
        }




def extract_topk_features(feat_map, center_map, K=100):
    """
    feat_map: [B, C, H, W]
    center_map: [B, 1, H, W]  (热力图)
    return: topk_feats [B, K, C], indices [B, K]
    """
    B, C, H, W = feat_map.shape
    center_flat = center_map.view(B, -1)                  # [B, H*W]
    scores, inds = torch.topk(center_flat, K, dim=1)      # [B, K]

    feats = []
    for b in range(B):
        feat_flat = feat_map[b].view(C, -1).permute(1, 0)  # [H*W, C]
        feat_sel = feat_flat[inds[b]]                     # [K, C]
        feats.append(feat_sel)
    feats = torch.stack(feats, dim=0)  # [B, K, C]
    return feats, inds
