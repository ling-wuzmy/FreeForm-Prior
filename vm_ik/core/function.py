import os.path as osp
import numpy as np
import cv2, os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import pickle
from itertools import islice
import json
import time
import scipy.sparse as ssp
import matplotlib.pyplot as plt

from vm_ik.core.base import prepare_network
from vm_ik.core.config import cfg
from vm_ik.utils.funcs_utils import lr_check, save_obj
from vm_ik.utils.vis import vis_joints_3d, render_mesh, denormalize_image
from vm_ik.utils.aug_utils import get_affine_transform, flip_img, augm_params

class Simple3DMeshTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()

    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            if master:
                print('=> Evaluating on ', dataset_name, ' ... ')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()

                    imgs = meta['img'].cuda()
                    batch_size = imgs.shape[0]
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()

                    gt_pose, gt_mesh = meta['joint_cam'].cuda(), meta['mesh_cam'].cuda()

                    _, pred_uvd_pose, _, confidence, pred_mesh, _, _,_,_ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_mask=None, is_train=False)

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())

                    # flip_test
                    if isinstance(imgs, list):
                        imgs_flip = [flip_img(img.clone()) for img in imgs]
                    else:
                        imgs_flip = flip_img(imgs.clone())

                    _, _, _, _, pred_mesh_flip, _, _,_,_ = self.model(imgs_flip, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=(pred_uvd_pose, confidence), flip_output=True, flip_mask=None, is_train=False)

                    pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_flip), torch.matmul(self.vm_B.T[None], pred_mesh_flip[:, self.selected_indices])), dim=1)

                    results['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    results['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())

                    j_error = val_dataset.compute_joint_err(pred_pose_flip, gt_pose)

                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => joint error: {j_error:.4f}')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i), nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_pred_flip.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_pred_flip.jpg'.format(dataset_name, i), nrow=min(batch_size//3, 4))

                        joint_error += j_error
                for term in results.keys():
                    results[term] = np.concatenate(results[term])
            
                self.joint_error = joint_error / max(len(val_loader),1)

                if master:
                    joint_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'], results['gt_pose'])
                    mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)

                    msg = ''
                    msg += f'\n{eval_prefix}'
                    for metric_key in joint_flip_error_dict.keys():
                        metric_dict[metric_key+'_REG'].append(joint_flip_error_dict[metric_key].item())
                        msg += f' | {metric_key:12}: {joint_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix}'
                    for metric_key in mesh_flip_error_dict.keys():
                        metric_dict[metric_key].append(mesh_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_flip_error_dict[metric_key]:3.2f}'
                    print(msg)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value[-1], epoch)

                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')

class Simple3DMeshPostTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        # initialize
        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()
        # noise_reduction
        assert cfg.model.simple3dmesh.noise_reduce, 'only support noise_reduce is True'


    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            print('=> Evaluating on ', dataset_name, ' ... ')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()
                    imgs = meta['img'].cuda()
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()

                    gt_pose, gt_mesh = meta['joint_cam'].cuda(), meta['mesh_cam'].cuda()

                    _, pred_uvd_pose, _, confidence, pred_mesh, pred_mesh_post, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_mask=None, is_train=False)

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())

                    # flip_test
                    if isinstance(imgs, list):
                        imgs_flip = [flip_img(img.clone()) for img in imgs]
                    else:
                        imgs_flip = flip_img(imgs.clone())

                    _, _, _, _, pred_mesh_flip, pred_mesh_post_flip, _ = self.model(imgs_flip, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=(pred_uvd_pose, confidence), flip_output=True, flip_mask=None, is_train=False)

                    pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_post_flip), torch.matmul(self.vm_B.T[None], pred_mesh_post_flip[:, self.selected_indices])), dim=1)

                    results['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                    results['pred_mesh_post_flip'].append(pred_mesh_post_flip.detach().cpu().numpy())
                    results['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())
    
                    j_error = val_dataset.compute_joint_err(pred_pose_flip, gt_pose)

                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => joint error: {j_error:.4f}')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind)
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_pred_flip.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind)
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh_post_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_pred_post_flip.jpg'.format(dataset_name, i))
                        joint_error += j_error

                for term in results.keys():
                    results[term] = np.concatenate(results[term])
            
                self.joint_error = joint_error / len(val_loader)

                if master:
                    joint_post_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'], results['gt_pose'])
                    mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)
                    mesh_post_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_post_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)

                    msg = ''
                    msg += f'\n{eval_prefix} post '
                    for metric_key in joint_post_flip_error_dict.keys():
                        metric_dict[metric_key+'_POST'].append(joint_post_flip_error_dict[metric_key].item())
                        msg += f' | {metric_key:12}: {joint_post_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix} '
                    for metric_key in mesh_flip_error_dict.keys():
                        metric_dict[metric_key].append(mesh_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix} post '
                    for metric_key in mesh_post_flip_error_dict.keys():
                        metric_dict[metric_key+'_POST'].append(mesh_post_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_post_flip_error_dict[metric_key]:3.2f}'

                    print(msg)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value[-1], epoch)

                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')
