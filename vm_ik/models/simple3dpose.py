from os import path as osp
import numpy as np

np.set_printoptions(suppress=True)
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import scipy.sparse as ssp
import pickle

from vm_ik.models.layers.HRnet import HRNet

from vm_ik.core.config import cfg, update_config
from vm_ik.models.transformer import *
from .position_encoding import build_position_encoding
from .layers.smpl.SMPL import SMPL_layer


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [B,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1) * cfg.model.simple3dpose.alpha
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


class Simple3DPose(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d,
                 flip_pairs=None):
        super(Simple3DPose, self).__init__()
        self.deconv_dim = cfg.model.simple3dpose.num_deconv_filters
        self._norm_layer = norm_layer
        self.joint_num = cfg.dataset.num_joints
        self.actual_joint_num = 81
        self.norm_type = cfg.model.simple3dpose.extra_norm_type
        self.depth_dim = cfg.model.simple3dpose.extra_depth_dim
        self.height_dim = cfg.model.heatmap_shape[0]
        self.width_dim = cfg.model.heatmap_shape[1]

        self.flip_pairs_left = [pair[0] for pair in flip_pairs] if flip_pairs is not None else None
        self.flip_pairs_right = [pair[1] for pair in flip_pairs] if flip_pairs is not None else None
        self.joint_pairs_29 = ((1, 2), (4, 5), (7, 8),
                               (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),
                               (22, 23), (25, 26), (27, 28))

        backbone = HRNet
        self.selected_indices = [i for i in range(6890)]
        with open(osp.join(cfg.data_dir, cfg.dataset.smpl_indices_path), 'rb') as f:
            smpl_indices = pickle.load(f)
        for body_part in smpl_indices.keys():
            body_part_indices = list(smpl_indices[body_part].numpy())
            if body_part in cfg.model.mesh2vm.ignore_part:
                for idx in body_part_indices:
                    self.selected_indices.remove(idx)
        self.smpl_dtype = torch.float32
        init_shape = np.load('./data/smpl/h36m_mean_beta.npy')
        self.register_buffer(
            'init_shape',
            torch.Tensor(init_shape).float())
        h36m_jregressor = np.load('./data/smpl/J_regressor_h36m_correct.npy')
        self.smpl = SMPL_layer(
            './data/smpl/SMPL_NEUTRAL.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=self.smpl_dtype
        )
        self.vm_B = torch.Tensor(
            ssp.load_npz(osp.join(cfg.model.mesh2vm.vm_path, f'vm_B{cfg.model.mesh2vm.vm_type}.npz')).toarray().astype(
                float)).cuda()
        self.preact = backbone()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decshape = nn.Linear(1024, 10)
        self.decphi = nn.Linear(1024, 23 * 2)  # [cos(phi), sin(phi)]
        num_enc_layers = 4
        encoder_norm11 = nn.LayerNorm(640)
        encoder_norm21 = nn.LayerNorm(128)
        encoder_layer11 = TransformerEncoderLayer(640, 8, 2048, 0.1, "relu")
        encoder_layer21 = TransformerEncoderLayer(128, 8, 2048, 0.1, "relu")
        self.transformer11 = TransformerEncoder(encoder_layer11, num_enc_layers // 2, None)
        self.transformer21 = TransformerEncoder(encoder_layer11, num_enc_layers // 2, encoder_norm11)
        self.transformer31 = TransformerEncoder(encoder_layer21, num_enc_layers // 2, None)
        self.transformer41 = TransformerEncoder(encoder_layer21, num_enc_layers // 2, encoder_norm21)
        self.token_position_embed1 = nn.Embedding(29 + 64, 128)
        self.dim_reduce1 = nn.Linear(640, 128)
        self.re_regressor1 = nn.Linear(128, 1)
        encoder_norm1 = nn.LayerNorm(640)
        encoder_norm2 = nn.LayerNorm(128)
        encoder_layer1 = TransformerEncoderLayer(640, 8, 2048, 0.1, "relu")
        encoder_layer2 = TransformerEncoderLayer(128, 8, 2048, 0.1, "relu")
        self.transformer1 = TransformerEncoder(encoder_layer1, num_enc_layers // 2, None)
        self.transformer2 = TransformerEncoder(encoder_layer1, num_enc_layers // 2, encoder_norm1)
        self.transformer3 = TransformerEncoder(encoder_layer2, num_enc_layers // 2, None)
        self.transformer4 = TransformerEncoder(encoder_layer2, num_enc_layers // 2, encoder_norm2)
        self.pos_embedding = build_position_encoding(hidden_dim=512)
        self.token_position_embed = nn.Embedding(81 + 64, 128)
        self.dim_reduce = nn.Linear(640, 128)
        self.re_regressor = nn.Linear(128, 1)

        self.root_idx = 0

    def _make_deconv_layer(self):
        deconv_layers = []
        if self.height_dim == 80:
            deconv1 = nn.ConvTranspose2d(
                self.feature_channel, self.deconv_dim[0], kernel_size=7, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn1 = self._norm_layer(self.deconv_dim[0])
            deconv2 = nn.ConvTranspose2d(
                self.deconv_dim[0], self.deconv_dim[1], kernel_size=6, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn2 = self._norm_layer(self.deconv_dim[1])
            deconv3 = nn.ConvTranspose2d(
                self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn3 = self._norm_layer(self.deconv_dim[2])
        else:
            deconv1 = nn.ConvTranspose2d(
                self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn1 = self._norm_layer(self.deconv_dim[0])
            deconv2 = nn.ConvTranspose2d(
                self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn2 = self._norm_layer(self.deconv_dim[1])
            deconv3 = nn.ConvTranspose2d(
                self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * cfg.model.input_shape[1]
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * cfg.model.input_shape[0]
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        uv_jts_clone = uv_jts[:, :1].clone()
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        # xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts, uv_jts_clone

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True, flip_mask=None):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.actual_joint_num, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        if flip_mask is None:
            flip_mask = torch.ones(num_batches).bool()
        # none of them needs flip
        if flip_mask.sum() == 0:
            return pred_jts
        flip_mask = flip_mask.cuda()

        # flip
        if shift:
            pred_jts[flip_mask, :, 0] = - pred_jts[flip_mask, :, 0]
        else:
            pred_jts[flip_mask, :, 0] = -1 / self.width_dim - pred_jts[flip_mask, :, 0]

        # flip_pair
        pred_jts_flip = pred_jts[flip_mask].clone()

        pred_jts_flip[:, self.flip_pairs_left], pred_jts_flip[:, self.flip_pairs_right] = \
            pred_jts_flip[:, self.flip_pairs_right].clone(), pred_jts_flip[:, self.flip_pairs_left].clone()

        pred_jts[flip_mask] = pred_jts_flip

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.actual_joint_num * 3)

        return pred_jts

    def flip_uvd_coord_j(self, pred_jts, shift=False, flatten=True):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, 29, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        # flip
        if shift:
            pred_jts[:, :, 0] = - pred_jts[:, :, 0]
        else:
            pred_jts[:, :, 0] = -1 / self.width_dim - pred_jts[:, :, 0]

        for pair in self.joint_pairs_29:
            dim0, dim1 = pair
            idx = torch.Tensor((dim0, dim1)).long()
            inv_idx = torch.Tensor((dim1, dim0)).long()
            pred_jts[:, idx] = pred_jts[:, inv_idx]

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, 29 * 3)

        return pred_jts

    def flip_confidence(self, confidence, flip_mask=None):
        num_batches = confidence.shape[0]
        if flip_mask is None:
            flip_mask = torch.ones(num_batches).bool()
        # none of them needs flip
        if flip_mask.sum() == 0:
            return confidence
        flip_mask = flip_mask.cuda()

        # flip_pair
        confidence_flip = confidence[flip_mask].clone()

        confidence_flip[:, self.flip_pairs_left], confidence_flip[:, self.flip_pairs_right] = \
            confidence_flip[:, self.flip_pairs_right].clone(), confidence_flip[:, self.flip_pairs_left].clone()

        confidence[flip_mask] = confidence_flip

        return confidence

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False,
                flip_mask=None):
        """Forward pass
        Inputs:
            x: image, size = (B, 3, 224, 224)
        Returns:
            pred_xyz_jts: camera 3d pose (joints + archetypes), size = (B, J+K, 3)
            confidence: confidence score for each body point in 3d pose, size = (B, J+K), for loss_{conf}
            pred_uvd_jts_flat: uvd 3d pose (joints + archetypes), size = (B, (J+K)*3), for loss_{pose}
        """
        batch_size = x.shape[0]

        x1, x2, x3 = self.preact(x)  # (b, 512, 8, 8)
        device = x.device
        position_embedding1 = self.token_position_embed1.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        position_embedding = self.token_position_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        feature_map = x2.view(batch_size, 512, -1)
        feature_map += self.pos_embedding(batch_size, 64, 64, device).flatten(start_dim=2)
        grid_feature = x3.view(batch_size, 512, -1)
        grid_feature += self.pos_embedding(batch_size, 8, 8, device).flatten(2)
        out1 = x1.reshape((x1.shape[0], 93, -1))
        out1 = norm_heatmap(self.norm_type, out1)
        assert out1.dim() == 3, out1.shape

        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out1, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out1.shape[:2], 1), dtype=torch.float, device=out1.device)

        heatmaps11 = out1 / out1.sum(dim=2, keepdim=True)
        # print('heatmaps1',heatmaps1.shape)
        heatmaps_j, _ = heatmaps11.split([29, 64], dim=1)

        sampled_feature_j = heatmaps_j @ feature_map.transpose(1, 2).contiguous()  # BX431XC
        sampled_feature_j = torch.cat(
            [grid_feature.transpose(1, 2).contiguous(), sampled_feature_j], 1)
        sampled_feature_j = torch.cat([sampled_feature_j, position_embedding1], 2)
        sampled_feature_j = sampled_feature_j.permute(1, 0, 2)
        sampled_feature_j = self.transformer11(sampled_feature_j)
        sampled_feature_j = self.transformer21(sampled_feature_j)
        sampled_feature_reduce_j = self.dim_reduce1(sampled_feature_j)
        sampled_feature_reduce_j = self.transformer31(sampled_feature_reduce_j)
        sampled_feature_reduce_j = self.transformer41(sampled_feature_reduce_j)
        _, jv_features_j = sampled_feature_reduce_j.split([64, sampled_feature_reduce_j.shape[0] - 64], dim=0)
        out_j = self.re_regressor1(jv_features_j).permute(1, 0, 2)

        heatmaps_j = heatmaps_j.reshape((heatmaps_j.shape[0], 29, self.height_dim, self.width_dim))  # B, J+K, D, H, W

        hm_x_j = heatmaps_j.sum((2))
        hm_y_j = heatmaps_j.sum((3))

        device = torch.device('cuda')

        hm_x_j = hm_x_j * torch.arange(float(hm_x_j.shape[-1])).to(device)
        hm_y_j = hm_y_j * torch.arange(float(hm_y_j.shape[-1])).to(device)

        coord_x_j = hm_x_j.sum(dim=2, keepdim=True)
        coord_y_j = hm_y_j.sum(dim=2, keepdim=True)
        coord_z_j = out_j

        # pred_uvd_jts_coord_j = torch.cat((coord_x_j, coord_y_j, coord_z_j), dim=2).clone()     # B, J+K, 3

        coord_x_j = coord_x_j / float(self.width_dim) - 0.5
        coord_y_j = coord_y_j / float(self.height_dim) - 0.5

        pred_uvd_jts_29 = torch.cat((coord_x_j, coord_y_j, coord_z_j), dim=2)
        # print('pred_uvd_jts_29',pred_uvd_jts_29.shape)
        if flip_output:
            pred_uvd_jts_29 = self.flip_uvd_coord_j(pred_uvd_jts_29, flatten=False, shift=True)

        pred_xyz_jts_29, pred_root_xy_img_29 = self.uvd_to_cam(pred_uvd_jts_29, trans_inv, intrinsic_param, joint_root,
                                                               depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts_29)) == 0, ('pred_xyz_jts_29', pred_xyz_jts_29)

        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape((batch_size, 29 * 3))

        pred_xyz_jts_29 = pred_xyz_jts_29 - pred_xyz_jts_29[:, self.root_idx, :].unsqueeze(1)

        x2 = self.avg_pool(x2)
        x2 = x2.view(x2.size(0), -1)
        init_shape = self.init_shape.expand(batch_size, -1)  # (B, 10,)

        xc = x2

        xc = self.fc1(xc)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)

        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape
        pred_phi = self.decphi(xc)
        pred_phi = pred_phi.reshape(batch_size, 23, 2)

        output = self.smpl.hybrik(
            pose_skeleton=pred_xyz_jts_29.type(self.smpl_dtype) * 2,
            betas=pred_shape.type(self.smpl_dtype),
            phis=pred_phi.type(self.smpl_dtype),
            global_orient=None,
            return_verts=True
        )
        pred_vertices = output.vertices.float()
        pred_xyz_jts_17 = output.joints_from_verts.float() / 2
        # print('pred_vertices',pred_vertices.shape)
        # print('pred_xyz_jts_17',pred_xyz_jts_17.shape)
        # print('vm_B',self.vm_B.T[None].shape)
        pred_pose = torch.cat(
            (pred_xyz_jts_17, torch.matmul(self.vm_B.T[None], pred_vertices[:, self.selected_indices])), dim=1)
        pred_pose_d = pred_pose[:, :, 2]
        _, heatmaps1 = heatmaps11.split([12, 81], dim=1)
        sampled_feature = heatmaps1 @ feature_map.transpose(1, 2).contiguous()
        sampled_feature = sampled_feature + pred_pose_d.unsqueeze(-1)
        # print('sampled_feature',sampled_feature.shape)
        sampled_feature = torch.cat(
            [grid_feature.transpose(1, 2).contiguous(), sampled_feature], 1)
        sampled_feature = torch.cat([sampled_feature, position_embedding], 2)
        sampled_feature = sampled_feature.permute(1, 0, 2)
        sampled_feature = self.transformer1(sampled_feature)
        sampled_feature = self.transformer2(sampled_feature)
        sampled_feature_reduce = self.dim_reduce(sampled_feature)
        sampled_feature_reduce = self.transformer3(sampled_feature_reduce)
        sampled_feature_reduce = self.transformer4(sampled_feature_reduce)
        _, jv_features_2 = sampled_feature_reduce.split([64, sampled_feature_reduce.shape[0] - 64], dim=0)
        # print('jv_features_2',jv_features_2.shape)
        out2 = self.re_regressor(jv_features_2).permute(1, 0, 2)
        # print('out2',out2)

        heatmaps1 = heatmaps1.reshape(
            (heatmaps1.shape[0], self.actual_joint_num, self.height_dim, self.width_dim))  # B, J+K, D, H, W

        hm_x = heatmaps1.sum((2))
        hm_y = heatmaps1.sum((3))

        device = torch.device('cuda')

        hm_x = hm_x * torch.arange(float(hm_x.shape[-1])).to(device)
        hm_y = hm_y * torch.arange(float(hm_y.shape[-1])).to(device)

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = out2

        pred_uvd_jts_coord = torch.cat((coord_x, coord_y, coord_z), dim=2).clone()  # B, J+K, 3

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)  # B, J+K, 3
        # print('pred_uvd_jts',pred_uvd_jts.shape)

        # NOTE that heatmap is (z, y, x) pred_uvd_jts is (x, y, z)
        pred_uv_jts_ind = (pred_uvd_jts_coord[..., 1] * self.height_dim + pred_uvd_jts_coord[..., 0]).unsqueeze(
            2).long()
        confidence1 = torch.gather(heatmaps1.view(*heatmaps1.shape[:2], -1), 2, pred_uv_jts_ind).squeeze(-1)  # B, J+K
        confidence = confidence1

        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_orig, confidence_orig = flip_item

            pred_uvd_jts = self.flip_uvd_coord(pred_uvd_jts, flatten=False, shift=True)
            confidence = self.flip_confidence(confidence)

            pred_uvd_jts = (pred_uvd_jts + pred_uvd_jts_orig.reshape(batch_size, self.actual_joint_num, 3)) / 2
            confidence = (confidence + confidence_orig) / 2

        pred_uvd_jts_flat = pred_uvd_jts.reshape((batch_size, self.actual_joint_num * 3)).clone()

        # use flip_mask to flip back thosed flipped
        if flip_mask is not None:
            pred_uvd_jts = self.flip_uvd_coord(pred_uvd_jts, flatten=False, shift=True, flip_mask=flip_mask)
            confidence = self.flip_confidence(confidence, flip_mask=flip_mask)

        #  -0.5 ~ 0.5
        # Rotate back
        pred_xyz_jts, pred_root_xy_img = self.uvd_to_cam(pred_uvd_jts, trans_inv, intrinsic_param, joint_root,
                                                         depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts)) == 0, ('pred_xyz_jts', pred_xyz_jts)

        pred_xyz_jts = pred_xyz_jts - pred_xyz_jts[:, self.root_idx, :].unsqueeze(1)  # B, J+K, 3

        # pred_xyz_jts = pred_xyz_jts.reshape((batch_size, -1))
        return pred_xyz_jts, confidence, pred_uvd_jts_flat, pred_root_xy_img, pred_uvd_jts_29_flat, pred_pose


def get_model(flip_pairs):
    model = Simple3DPose(flip_pairs=flip_pairs)
    return model