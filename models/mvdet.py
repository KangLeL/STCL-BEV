import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from kornia.geometry.transform.imgwarp import warp_perspective
from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18, Encoder_eff, Encoder_swin_t, Encoder_res34
from models.decoder import Decoder

from .deformerable import DeformTransWorldFeat, create_reference_map


class MVDet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 num_cameras=None,
                 num_ids=None,
                 latent_dim=512,
                 encoder_type='res18',
                 device=torch.device('cuda'),
                 use_deformable=False,
                 use_Temporal=False,
                 use_new_deformable=False,
                 max_history=0,
                 use_early_fusion=False,
                 **kwargs
                 ):
        super().__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'res34', 'effb0', 'effb4', 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip

        if use_deformable:
            latent_dim = 128

        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras

        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.feat2d_dim = 128
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(self.feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(self.feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(self.feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(self.feat2d_dim)
        elif encoder_type == 'res34':
            self.encoder = Encoder_res34(self.feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(self.feat2d_dim)
        else:
            self.encoder = Encoder_eff(self.feat2d_dim, version='b4')

        if self.num_cameras is None:
            self.world_conv = nn.Sequential(
                nn.Conv3d(self.feat2d_dim, self.feat2d_dim, 3, padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                # nn.Conv3d(latent_dim, latent_dim, 3, padding=(1, 2, 2), dilation=(1, 2, 2)),
            )
        else:
            self.world_feat = nn.Sequential(
                nn.Conv2d(self.feat2d_dim * self.num_cameras, latent_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(latent_dim), nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            )

        # self.cam_out = nn.Sequential(
        #     nn.Conv2d(latent_dim, self.feat2d_dim, 3, padding=1),
        #     nn.InstanceNorm2d(self.feat2d_dim), nn.ReLU(),
        #     nn.Conv2d(self.feat2d_dim, self.feat2d_dim, 1, padding=0),
        # )

        kwargs['Y'] = Y
        kwargs['X'] = X

        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=2,
            n_ids=num_ids,
            **kwargs
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.use_deformable = use_deformable
        self.use_Temporal = use_Temporal

        # temporal
        self.pre_world_feat = None
        self.pre_index = None
        if max_history > 0:
            self.new_fusion_temporal = TemporalFusionModule(latent_dim, latent_dim)
            self.history_queue = []
        self.max_history = max_history
        # deformable
        if use_deformable:
            downsample = 2
            n_points = 4
            self.deformerable = DeformTransWorldFeat(num_cameras, (Y, X), self.feat2d_dim, hidden_dim=latent_dim,
                                                     n_points=n_points,
                                                     stride=downsample, use_Temporal=use_Temporal,
                                                     use_new_deformable=use_new_deformable,use_early_fusion=use_early_fusion)

    def add_history(self, bev_feat):
        if len(self.history_queue) >= self.max_history:
            self.history_queue.pop(0)
        self.history_queue.append(bev_feat)

    def clear_history(self):
        self.history_queue = []

    def get_history_stack(self):
        if len(self.history_queue) == 0:
            return None
        return torch.stack(self.history_queue, dim=1)

    def reset(self):
        self.decoder.reset()
        self.clear_history()

    def multi_view_to_bev(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, pre_world_feat=None):
        """
        将多视角图像融合为全局 BEV 特征
        输入:
            rgb_cams: (B,S,3,H,W)
            pix_T_cams: (B,S,4,4)
            cams_T_global: (B,S,4,4)
            vox_util: vox util object
            ref_T_global: (B,4,4)
        输出:
            world_features: (B, latent_dim, Y, X) BEV 特征
        """
        B, S, C, H, W = rgb_cams.shape
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)

        # --- 预处理 ---
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams_ = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams_)  # B*S,4,4

        # --- RGB Encoder ---
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,Cf,Hf,Wf
        _, Cf, Hf, Wf = feat_cams_.shape

        # --- intrinsics 缩放 ---
        sy = Hf / float(H)
        sx = Wf / float(W)
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)  # B*S,4,4

        # --- 投影矩阵 ---
        featpix_T_ref_ = torch.matmul(featpix_T_cams_[:, :3, :3], cams_T_ref_[:, :3, [0, 1, 3]])  # B*S,3,3
        ref_T_mem = vox_util.get_ref_T_mem(B, self.Y, self.Z, self.X)  # B,4,4
        ref_T_mem = ref_T_mem[0, [0, 1, 3]][:, [0, 1, 3]]  # 3,3
        featpix_T_mem_ = torch.matmul(featpix_T_ref_, ref_T_mem)  # B*S,3,3
        mem_T_featpix = torch.inverse(featpix_T_mem_)  # B*S,3,3
        proj_mats = mem_T_featpix

        # --- 多视角投影到 BEV ---
        world_features_ = warp_perspective(feat_cams_, proj_mats, (self.Y, self.X), align_corners=False)
        world_features = __u(world_features_)  # B,S,Cf,Y,X

        # --- 融合不同相机 ---
        if self.use_deformable:
            world_features = self.deformerable(world_features, pre_world_feat)
        elif self.num_cameras is None:
            world_features = self.world_conv(world_features.permute(0, 2, 1, 3, 4))
            world_features = self.world_feat(world_features.sum(2, keepdim=True))
        else:
            world_features = self.world_feat(world_features.view(B, S * self.feat2d_dim, self.Y, self.X))

        return world_features, feat_cams_

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global,
                history_imgs=None, history_intrins=None, history_extrins=None, valid_bev=None, index=None):

        if self.use_Temporal:
            if self.pre_index is not None and (index - self.pre_index).item() != 1:
                self.pre_world_feat = None
            # elif self.pre_world_feat is not None:
            #     self.pre_world_feat = self.pre_world_feat * valid_bev
        if self.max_history > 0:
            if self.pre_index is not None and (index - self.pre_index).item() != 1:
                self.clear_history()
        world_features, feat_cams_ = self.multi_view_to_bev(
            rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, self.pre_world_feat
        )
        if self.max_history > 0:
            self.add_history(world_features.detach().clone())
            history_bev_feat = self.get_history_stack()
            world_features = self.new_fusion_temporal(history_bev_feat)
        if self.use_Temporal:
            self.pre_world_feat = world_features.detach().clone()
        self.pre_index = index

        # get history BEV features
        history_bev = []
        if history_imgs:
            with torch.no_grad():
                for history_img, history_intrin, history_extrin in zip(history_imgs, history_intrins, history_extrins):
                    history_world_feature, _ = self.multi_view_to_bev(
                        history_img, history_intrin, history_extrin, vox_util, ref_T_global
                    )
                    history_bev.append(history_world_feature)

        out_dict = self.decoder(
            world_features,
            feat_cams_,
            history_bev=history_bev,
            pre_valid_bev=valid_bev
        )
        return out_dict


class TemporalFusionModule(nn.Module):
    """
    时序融合模块：使用ConvGRU聚合时序信息。
    输入: 一个时序序列的BEV特征 [B, T, C, H, W]
    输出: 当前帧经过时序融合后的BEV特征 [B, C, H, W]
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        # 使用ConvGRU细胞单元
        self.conv_gru_cell = ConvGRUCell(input_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels
        # 可选的：一个输出卷积，用于进一步处理GRU输出的隐藏状态
        self.output_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

    def forward(self, x):
        # x.shape: [Batch, Time, Channels, Height, Width]
        b, t, c, h, w = x.size()

        # 初始化隐藏状态（对于每个样本初始化为0）
        hidden_state = torch.zeros(b, self.hidden_channels, h, w).to(x.device)

        # 按时间步循环，逐步更新隐藏状态
        for time_step in range(t):
            hidden_state = self.conv_gru_cell(x[:, time_step, :, :, :],  # 当前时间步的BEV特征
                                              hidden_state)  # 上一时间步的隐藏状态

        # 最终的隐藏状态包含了整个序列的融合信息
        output = self.output_conv(hidden_state)
        return output  # 输出形状: [B, C, H, W]


class ConvGRUCell(nn.Module):
    """ 实现一个卷积GRU细胞单元 """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        # 确保卷积的padding能保持特征图尺寸不变
        padding = kernel_size // 2

        # 重置门（Reset Gate）r_t
        self.conv_reset = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        # 更新门（Update Gate）z_t
        self.conv_update = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        # 候选隐藏状态（Candidate Hidden State）~h_t
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

        # 初始化门控卷积的权重
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h_prev):
        # x: 当前输入, h_prev: 上一个隐藏状态
        # 将输入和上一隐藏状态在通道维拼接
        combined = torch.cat([x, h_prev], dim=1)  # dim=1 是通道维

        # 计算重置门和更新门
        reset_gate = torch.sigmoid(self.conv_reset(combined))
        update_gate = torch.sigmoid(self.conv_update(combined))

        # 将重置门应用于上一个隐藏状态
        h_reset = reset_gate * h_prev
        # 拼接输入和重置后的隐藏状态
        combined_candidate = torch.cat([x, h_reset], dim=1)
        # 计算候选隐藏状态
        candidate = torch.tanh(self.conv_candidate(combined_candidate))

        # 计算新的隐藏状态: h_t = (1 - z_t) * h_prev + z_t * ~h_t
        h_new = (1 - update_gate) * h_prev + update_gate * candidate

        return h_new
