#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import yaml
import random
import os
import pickle
import math
import torch
from torch.utils import data
from torchvision.transforms import transforms
from PIL import Image

from dataloader.dataset import spherical_dataset, nb_process_inst, nb_process_label
from dataloader.pie_aug_utils import proj_lidar2img, merge_images_pitch_np, fit_to_box, merge_images_yaw, expand_box, crop_box_img, paste_box_img, fit_box_cv, draw_dashed_box

class Nuscenes_pt_ial(data.Dataset):
    def __init__(self, data_path, split, cfgs, nusc, version, assync_compensation=True):
        with open("nuscenes.yaml", 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        sample_pkl_path = cfgs['dataset']['sample_pkl_path']

        if version == 'v1.0-mini':
            if split == 'train':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_val_mini_ial.pkl")
            elif split == 'val':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_val_mini_ial.pkl")
        elif version == 'v1.0-trainval':
            if split == 'train':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_train.pkl")
            elif split == 'val':
                imageset = os.path.join(sample_pkl_path, "nuscenes_infos_val.pkl")
        elif version == 'v1.0-test':
            imageset = os.path.join(sample_pkl_path, "nuscenes_infos_test.pkl")
        else:
            raise NotImplementedError

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        self.learning_map = nuscenesyaml['learning_map']
        self.split = split
        self.thing_list = [cl for cl, is_thing in nuscenesyaml['thing_class'].items() if is_thing]
        self.nusc_infos = data['infos']
        assert 'images' in data['infos'][0].keys() # load lidar2image info
        self.data_path = data_path
        self.cfgs = cfgs
        self.nusc = nusc
        self.version = version

        # 多模态
        self.pix_fusion = self.cfgs['model']['pix_fusion']
        self.IMAGE_SIZE = (900, 1600)
        self.transform = transforms.Compose([transforms.Resize(size=[int(x * 0.4) for x in self.IMAGE_SIZE])])
        self.CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.open_asynchronous_compensation = assync_compensation
        # corresponding to self.CAM_CHANNELS，fov from https://www.nuscenes.org/nuscenes
        # 6 * 3 (cosine lowerbound, cosine upperbound, if_front)
        self.cam_fov = [[-np.cos(11 * math.pi/36), np.cos(11 * math.pi/36), 1], # CAM_FRONT
                        [np.cos(7 * math.pi /18), 1, 1], # CAM_FRONT_RIGHT
                        [-1, -np.cos(7 * math.pi / 18), 1], # CAM_FRONT_LEFT
                        [-0.5, 0.5, -1], # CAM_BACK 120 degrees fov
                        [-1, -np.cos(7 * math.pi /18), -1], # CAM_BACK_LEFT
                        [np.cos(7 * math.pi / 18), 1, -1]] #CAM_BACK_RIGHT 

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        if self.version == "v1.0-trainval":
            lidar_path = info['lidar_path'][16:]
        elif self.version == "v1.0-mini":
            lidar_path = info['lidar_path'][44:]
        elif self.version == "v1.0-test":
            lidar_path = info['lidar_path'][16:]
            
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        
        data_tuple = (points[:, :3], points[:, 3], lidar_sd_token)
        
        # load label
        if self.version != "v1.0-test":
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            panoptic_labels_filename = os.path.join(self.nusc.dataroot,
                                                    self.nusc.get('panoptic', lidar_sd_token)['filename'])
            panoptic_label = np.load(panoptic_labels_filename)['data']
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
            noise_mask = points_label == 0
            points_label[noise_mask] = 17
            data_tuple += (points_label.astype(np.uint8), panoptic_label)
        else:
            data_tuple += (-1, -1)

        if self.pix_fusion:
            # load lidar2image projection matrix
            lidar2cams, cam2imgs, camera_channel = [], [], []
            for v in self.CAM_CHANNELS:
                lidar2cams.append(self.nusc_infos[index]['images'][v]['lidar2cam'])
                cam2imgs.append(self.nusc_infos[index]['images'][v]['cam2img'])
                im = Image.open(os.path.join(self.data_path, 'samples', v, self.nusc_infos[index]['images'][v]['img_path'])).convert('RGB')
                camera_channel.append(np.array(self.transform(im)).astype('float32'))
                ori_size = (im.size[0], im.size[1])
            lidar2imgs = pack_lidar2imgs(lidar2cams, cam2imgs)
            ori_camera_channel = np.stack(camera_channel, axis=0)
            for i in range(6):
                camera_channel[i] /= 255.0
                camera_channel[i][:, :, 0] = (camera_channel[i][:, :, 0] - 0.485) / 0.229
                camera_channel[i][:, :, 1] = (camera_channel[i][:, :, 1] - 0.456) / 0.224
                camera_channel[i][:, :, 2] = (camera_channel[i][:, :, 2] - 0.406) / 0.225
            camera_channel = np.stack(camera_channel, axis=0)
            fusion_tuple = (camera_channel, lidar2imgs, ori_camera_channel, ori_size)
            
            data_tuple += (fusion_tuple,)
        return data_tuple


class spherical_dataset_pie(spherical_dataset):
    def __init__(self, in_dataset, cfgs, ignore_label=0, fixed_volume_space=True, use_aug=True):
        'Initialization'
        self.point_cloud_dataset = in_dataset
    
        # Initialize LaserMix parameters
        self.use_laser = cfgs['dataset']['pie_aug'].get('use_laser', False)
        if self.use_laser:
            self.laser_mix_prob = cfgs['dataset']['pie_aug'].get('laser_mix_prob', 0.1)
            self.num_areas = random.choice(cfgs['dataset']['pie_aug'].get('num_areas', [3,4,5,6]))
            assert isinstance(self.num_areas, int)
            self.pitch_angles = cfgs['dataset']['pie_aug'].get('pitch_angles', [-30, 10])
        self.use_polar = cfgs['dataset']['pie_aug'].get('use_polar', False)
        if self.use_polar:
            self.polar_mix_prob = cfgs['dataset']['pie_aug'].get('polar_mix_prob', 0.1)
            self.instance_classes = cfgs['dataset']['pie_aug'].get('instance_classes', [1, 2, 3, 4, 5, 6, 7, 8])
            self.swap_ratio = cfgs['dataset']['pie_aug'].get('swap_ratio', 0.1)
            self.rotate_paste_ratio = cfgs['dataset']['pie_aug'].get('rotate_paste_ratio', 0.1)
        super(spherical_dataset_pie, self).__init__(
            in_dataset=in_dataset, 
            cfgs=cfgs, 
            ignore_label=ignore_label, 
            fixed_volume_space=fixed_volume_space, 
            use_aug=use_aug
        )
            

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 5:
            xyz, feat, token, labels, insts = data
            fusion_tuple = None
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        elif len(data) == 6:
            xyz, feat, token, labels, insts, fusion_tuple = data
            if len(feat.shape) == 1: feat = feat[..., np.newaxis]
        else:
            raise Exception('Return invalid data tuple')
        
        if type(labels)==np.ndarray:
            if len(labels.shape) == 1: labels = labels[..., np.newaxis]
            if len(insts.shape) == 1: insts = insts[..., np.newaxis]

        # Apply LaserMix augmentation
        if self.use_laser or self.use_polar:
            # Select another random point cloud for mixing
            mix_index = random.randint(0, len(self.point_cloud_dataset) - 1)
            mix_data = self.point_cloud_dataset[mix_index]
            if not fusion_tuple:
                mix_xyz, mix_feat, _, mix_labels, mix_insts = mix_data
            else:
                camera_channel, lidar2imgs, ori_camera_channel, ori_size = fusion_tuple
                img_scale = camera_channel.shape[2] / ori_size[0]
                mix_xyz, mix_feat, _, mix_labels, mix_insts, mix_fusion_tuple = mix_data
                mix_camera_channel, mix_lidar2imgs, _, _ = mix_fusion_tuple
            if type(mix_labels)==np.ndarray:
                if len(mix_labels.shape) == 1: mix_labels = mix_labels[..., np.newaxis]
                if len(mix_insts.shape) == 1: mix_insts = mix_insts[..., np.newaxis]
            if len(mix_feat.shape) == 1: mix_feat = mix_feat[..., np.newaxis]
                
            if self.use_laser and np.random.random() < self.laser_mix_prob:
                xyz, labels, insts, feat, camera_channel = self.laser_mix(xyz, labels, insts, feat, 
                                                      mix_xyz, mix_labels, mix_insts, mix_feat,
                                                      (fusion_tuple is not None), 
                                                      camera_channel, mix_camera_channel, mix_lidar2imgs, 
                                                      ori_size, img_scale)
            if self.use_polar and np.random.random() < self.polar_mix_prob:
                xyz, labels, insts, feat, camera_channel = self.polar_mix(xyz, labels, insts, feat, 
                                                      mix_xyz, mix_labels, mix_insts, mix_feat,
                                                      (fusion_tuple is not None), 
                                                      camera_channel, mix_camera_channel, mix_lidar2imgs, 
                                                      ori_size, img_scale)
        ############################## Preprocess Part ##############################
        
        # 转化成极坐标系
        xyz_pol = cart2polar(xyz)

        # 统一使用预先定义好的坐标范围
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # 把点转换成其对应的网格坐标，其中加一个1e-8是为了clip后的点，边界上不会越界
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size)
        min_bound = min_bound + [1e-8, 1e-8, 1e-8]
        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(int)

        # 每个网格的起始角落在真实坐标系下的绝对位置
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        if type(labels)==np.ndarray and type(insts)==np.ndarray:
            # 生成每个voxel的语义label，单个网格内采用最大投票
            voxel_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            current_grid = grid_ind[:np.size(labels)]
            label_voxel_pair = np.concatenate([current_grid, labels], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((current_grid[:, 0], current_grid[:, 1], current_grid[:, 2])), :]
            voxel_label = nb_process_label(np.copy(voxel_label), label_voxel_pair)

            # 生成前景点的mask，insts为0的点要单独特判一下，可能是有个别前景物体的点被标记为了insts为0
            mask = np.zeros_like(labels, dtype=bool)
            for label in self.point_cloud_dataset.thing_list:
                # mask[labels == label] = True
                mask[np.logical_and(labels == label, insts != 0)] = True

            # 生成每个voxel的实例insts id，单个网格内采用最大投票
            voxel_inst = insts[mask].squeeze()
            unique_inst = np.unique(voxel_inst)
            unique_inst_dict = {label: idx + 1 for idx, label in enumerate(unique_inst)}
            if voxel_inst.size > 1:
                voxel_inst = np.vectorize(unique_inst_dict.__getitem__)(voxel_inst)
                # process panoptic
                processed_inst = np.ones(self.grid_size[:2], dtype=np.uint8) * self.ignore_label
                inst_voxel_pair = np.concatenate([current_grid[mask[:, 0], :2], voxel_inst[..., np.newaxis]], axis=1)
                inst_voxel_pair = inst_voxel_pair[np.lexsort((current_grid[mask[:, 0], 0], current_grid[mask[:, 0], 1])), :]
                processed_inst = nb_process_inst(np.copy(processed_inst), inst_voxel_pair)
            else:
                # processed_inst = np.zeros([480, 360])
                processed_inst = np.zeros([self.grid_size[0], self.grid_size[1]])

            center, center_points, offset = self.panoptic_proc(insts[mask], xyz[:np.size(labels)][mask[:, 0]],
                                                            processed_inst, voxel_position[:2, :, :, 0],
                                                            unique_inst_dict, min_bound, intervals)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)
        return_fea = np.concatenate((return_xyz, feat), axis=1)

        # bev_mask = np.zeros((1, 480, 360), dtype=bool)
        bev_mask = np.zeros((1, self.grid_size[0], self.grid_size[1]), dtype=bool)
        uni_out = np.unique(grid_ind[:,0:2],axis=0)
        bev_mask[0, uni_out[:,0], uni_out[:,1]] = True

        return_dict = {}
        return_dict['lidar_token'] = token
        return_dict['xyz_cart'] = xyz
        return_dict['return_fea'] = return_fea
        return_dict['pol_voxel_ind'] = grid_ind
        # return_dict['rotate_deg'] = rotate_deg
        
        if type(labels) == np.ndarray and type(insts) == np.ndarray:
            return_dict['voxel_label'] = voxel_label
            return_dict['gt_center'] = center
            return_dict['gt_offset'] = offset
            return_dict['inst_map_sparse'] = processed_inst != 0
            return_dict['bev_mask'] = bev_mask
            return_dict['pt_sem_label'] = labels
            return_dict['pt_ins_label'] = insts

        if len(data) == 6:
            # lidar2img projection
            # camera_channel, lidar2imgs, ori_camera_channel, ori_size = fusion_tuple
            # valid_mask: -1 for points not projected to any image, 0-5 for points projected to corresponding image
            # masks: mask for each image, True for points projected to the image
            points_projs, masks, valid_mask = tranfrom_worldc_to_camc_light(xyz, lidar2imgs, ori_size)
            points_projs = np.stack(points_projs, axis=0)
            masks = np.stack(masks, axis=0)
            return_dict['camera_channel'] = camera_channel
            return_dict['pixel_coordinates'] = points_projs
            return_dict['masks'] = masks
            return_dict['valid_mask'] = valid_mask
            return_dict['ori_camera_channel'] = ori_camera_channel
            if type(labels) == np.ndarray and type(insts) == np.ndarray:
                point_with_pix_mask = return_dict['valid_mask'] > -1
                # image labels projected by points
                return_dict['im_label'] = return_dict['pt_sem_label'][point_with_pix_mask]

        return return_dict

    def laser_mix(self, xyz, labels, insts, feat, mix_xyz, mix_labels, mix_insts, mix_feat, 
                  img_sync=True, img=None, mix_img=None, mix_l2i=None, ori_size=None, img_scale=None):
        # Get the pitch angle
        rho = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        pitch = np.arctan2(xyz[:, 2], rho)
        pitch = np.clip(pitch, self.pitch_angles[0] + 1e-5,
                        self.pitch_angles[1] - 1e-5)

        mix_rho = np.sqrt(mix_xyz[:, 0]**2 + mix_xyz[:, 1]**2)
        mix_pitch = np.arctan2(mix_xyz[:, 2], mix_rho)
        mix_pitch = np.clip(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)
    
        # Divide into regions and mix
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0], num=self.num_areas + 1)
        out_points, out_labels, out_insts, out_feats = [], [], [], []
        out_points_mix = []
        for i in range(self.num_areas):
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:
                idx = (pitch > start_angle) & (pitch <= end_angle)
                out_points.append(xyz[idx])
                out_labels.append(labels[idx])
                out_insts.append(insts[idx])
                out_feats.append(feat[idx])
            else:
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                out_points.append(mix_xyz[idx])
                out_points_mix.append(mix_xyz[idx])
                out_labels.append(mix_labels[idx])
                out_insts.append(mix_insts[idx])
                out_feats.append(mix_feat[idx])

        xyz = np.concatenate(out_points, axis=0)
        labels = np.concatenate(out_labels, axis=0)
        insts = np.concatenate(out_insts, axis=0)
        feat = np.concatenate(out_feats, axis=0)
        
        if img_sync:
            out_im_list = []
            image_list_orig = [img[i] for i in range(img.shape[0])]
            image_list_mix = [mix_img[i] for i in range(mix_img.shape[0])]
            
            for v in range(len(image_list_orig)):
                im_orig, im_mix = image_list_orig[v].copy(), image_list_mix[v].copy()
                for i in range(len(out_points_mix)):
                    points_img_b, mask_b = proj_lidar2img(out_points_mix[i], mix_l2i[v], 
                                                        img_size=ori_size,
                                                        min_dist=1.0)
                    points_img_b = points_img_b * img_scale
                    mixed_img, block = merge_images_pitch_np(im_orig, im_mix, points_img_b)  
                    im_orig = mixed_img
                ########## save image for debug ##########
                # import cv2
                # # sample_token = input_dict['token']
                # # view = list(input_dict['images'].keys())[v]
                # denorm_img = (mixed_img - mixed_img.min()) / (mixed_img.max() - mixed_img.min()) * 255
                # denorm_img = denorm_img.astype(np.uint8)
                # cv2.imwrite(f'misc/rendered_point_mmaug/laser_{v}_{i}.png', denorm_img)
                ########## save image for debug ##########
                out_im_list.append(im_orig.reshape(1, im_orig.shape[0], im_orig.shape[1], im_orig.shape[2]))
            return xyz, labels, insts, feat, np.concatenate(out_im_list, axis=0)
        
        return xyz, labels, insts, feat
    
    def polar_mix(self, xyz, labels, insts, feat, mix_xyz, mix_labels, mix_insts, mix_feat, 
                  img_sync=True, img=None, mix_img=None, mix_l2i=None, ori_size=None, img_scale=None):
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
            mix_yaw = -np.arctan2(mix_xyz[:, 1], mix_xyz[:,0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            a, b = xyz[idx].shape[0], mix_xyz[mix_idx].shape[0]
            block_orig, block_swap = xyz[idx], mix_xyz[mix_idx]
            block_sem_orig, block_sem_swap = labels[idx], mix_labels[mix_idx]
            block_inst_orig, block_inst_swap = insts[idx], mix_insts[mix_idx]
            block_feat_orig, block_feat_swap = feat[idx], mix_feat[mix_idx]
            swap_points = np.concatenate((block_orig, block_swap), axis=0)
            swap_pts_semantic_mask = np.concatenate((block_sem_orig, block_sem_swap), axis=0)
            swap_pts_instance_mask = np.concatenate((block_inst_orig, block_inst_swap), axis=0)
            swap_pts_feature = np.concatenate((block_feat_orig, block_feat_swap), axis=0)
            xyz = swap_points
            labels = swap_pts_semantic_mask
            insts = swap_pts_instance_mask
            feat = swap_pts_feature
        
            ########## image augmentation for yaw swap ##########
            if img_sync:
                out_im_list = []
                image_list_orig = [img[i] for i in range(img.shape[0])]
                image_list_mix = [mix_img[i] for i in range(mix_img.shape[0])]
        
                for v in range(len(image_list_orig)):
                    im_orig, im_mix = image_list_orig[v].copy(), image_list_mix[v].copy()
                    block_points_img, mask = proj_lidar2img(block_swap, mix_l2i[v], 
                                                            img_size=ori_size,
                                                            min_dist=1.0)
                    block_points_img = block_points_img * img_scale
                    l, r = fit_to_box(block_points_img, 'vertical')
                    img_new = merge_images_yaw(im_orig, im_mix, l, r)
                    
                    ########## save image for debug ##########
                    # sample_token = input_dict['token']
                    # view = list(input_dict['images'].keys())[v]
                    import cv2
                    denorm_img = (img_new - img_new.min()) / (img_new.max() - img_new.min()) * 255
                    denorm_img = denorm_img.astype(np.uint8)
                    cv2.imwrite(f'misc/rendered_point_mmaug/polar_swap_{v}.png', denorm_img)
                    ########## save image for debug ##########
                    out_im_list.append(img_new.reshape(1, img_new.shape[0], img_new.shape[1], img_new.shape[2]))
                img = np.concatenate(out_im_list, axis=0)
            ########## image augmentation ##########
            
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            add_points, add_sem, add_ins, add_feat = [], [], [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_labels == instance_class
                add_points.append(mix_xyz[mix_idx[:, 0], :])
                add_sem.append(mix_labels[mix_idx])
                add_ins.append(mix_insts[mix_idx])
                add_feat.append(mix_feat[mix_idx])
            
            add_points = np.concatenate(add_points, axis=0)
            add_sem = np.concatenate(add_sem, axis=0)
            add_ins = np.concatenate(add_ins, axis=0)
            add_feat = np.concatenate(add_feat, axis=0)
            
            # # rotate-copy
            copy_points = [add_points]
            copy_sem = [add_sem]
            copy_ins = [add_ins]
            copy_feat = [add_feat]
            block_cp = copy_points.copy()
            block_cp_instlabel = copy_ins.copy()
            block_cp_semlabel  = copy_sem.copy()
                
            angle_list = [
                # angle * np.pi,
                # 0.5 * np.pi * 2 / 3,
                # np.random.random() * np.pi * 2 / 3,
                # (np.random.random() + 1) * np.pi * 2 / 3
            ]
            #TODO-YINING: implement rotate in the future
            # print(f'$$$$$$$$$$$$$$$$ Warning: instance rotate has not implemented yet!!!! ')
            # for angle in angle_list:
            #     # print(f'instance rotate angle at: {angle}, {angle/np.pi*180}')
            #     new_points = instance_points.clone()
            #     # new_points.rotate(angle, axis=0)
            #     copy_points.append(new_points)
            #     copy_sem.append(instance_pts_semantic_mask)
            #     if mix_panoptic:
            #         copy_ins.append(instance_pts_instance_mask)
            #     if mix_sam_pred:
            #         copy_pts_sam_pmask_2d.append(instance_pts_sam_pmask_2d)
            #         copy_pts_sam_pscore_2d.append(instance_pts_sam_pscore_2d)
            #     if mix_tgt_pl:
            #         copy_pl_psem.append(instance_pl_psem)
            #         copy_pl_pins.append(instance_pl_pins)
            #         copy_pl_pscore.append(instance_pl_pscore)
            #     block_ro.append(new_points)
            #     block_ro_instlabel.append(instance_pts_instance_mask.copy())
            #     block_ro_semlabel.append(instance_pts_semantic_mask.copy())
                
            copy_points = np.concatenate(copy_points, axis=0)
            copy_sem = np.concatenate(copy_sem, axis=0).reshape(-1, 1)
            copy_ins = np.concatenate(copy_ins, axis=0).reshape(-1, 1)
            copy_feat = np.concatenate(copy_feat, axis=0).reshape(-1, 1)
            
            xyz = np.concatenate((xyz, copy_points), axis=0)
            labels = np.concatenate((labels, copy_sem), axis=0)
            insts = np.concatenate((insts, copy_ins), axis=0)
            feat = np.concatenate((feat, copy_feat), axis=0)
            ########## image augmentation for instance copy ##########
            #TODO-YINING: only support copy for now
            if img_sync:
                image_list_orig = [img[i].copy() for i in range(img.shape[0])]
                image_list_mix = [mix_img[i].copy() for i in range(mix_img.shape[0])]
                cp_coord = np.concatenate(block_cp, axis=0)
                cp_instlabel = np.concatenate(block_cp_instlabel) # at least know the number of cp
                cp_semlabel  = np.concatenate(block_cp_semlabel)
                N_cp = cp_instlabel.shape[0]
                
                if N_cp > 0:
                    out_im_list = []
                    for v in range(len(image_list_orig)):
                        im_orig = image_list_orig[v]
                        im_1_copy = im_orig.copy()
                        im_mix = image_list_mix[v]
                        cp_coord_img, mask = proj_lidar2img(cp_coord, mix_l2i[v], 
                                                            img_size=ori_size,
                                                            min_dist=1.0)
                        cp_coord_img = cp_coord_img * img_scale
                        cp_instlabel_v = cp_instlabel[mask]
                        cp_semlabel_v  = cp_semlabel[mask]
                        
                        for inst_id in np.unique(cp_instlabel_v):
                            inst_coord = cp_coord_img[cp_instlabel_v == inst_id]
                            inst_cls = cp_semlabel_v[cp_instlabel_v == inst_id][0]
                            box = fit_box_cv(inst_coord)
                            ##1.  expand the box
                            # box = expand_box_proportional(box, 0.2, im_2_np.shape[:2])
                            box_ex = expand_box(box, 10, im_mix.shape[:2])
                            ##2.  crop
                            crop_img = crop_box_img(box_ex, im_mix)
                            ##3.paste to orig image
                            im_1_copy = paste_box_img(box_ex, im_orig, crop_img)
                            ########## save image for debug ##########
                            # ##3.  draw the box
                            # import cv2
                            # im_save = im_1_copy.copy()
                            # im_save = draw_dashed_box(im_save, box_ex, thickness=10, dash_length=5)
                            # # sample_token = input_dict['token']
                            # # view = list(input_dict['images'].keys())[i]
                            # denorm_img = (im_save - im_save.min()) / (im_save.max() - im_save.min()) * 255
                            # denorm_img = denorm_img.astype(np.uint8)
                            # cv2.imwrite(f'misc/rendered_point_mmaug/polar_inst_{v}.png', denorm_img)
                            ########## save image for debug ##########

                        # plot_point_in_camview_2(cp_coord_img, coloring, im_1_copy, dot_size=3)
                        out_im_list.append(im_1_copy)#.reshape(1, im_1_copy.shape[0], im_1_copy.shape[1], im_1_copy.shape[2]))
                    img = np.stack(out_im_list, axis=0)
            ########## image augmentation ##########
        return  xyz, labels, insts, feat, img
        
        
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    fea = input_xyz[:, 2]
    return np.stack((rho, phi, fea), axis=1)

def get_lidar2img(lidar2cam, cam2img):
    """
    Get transform matrix that transform coordinate from lidar coord system to image coord system.
    Args:
        lidar2cam: [4, 4]
        cam2img: [3, 3]
    """

    assert lidar2cam.shape == (4, 4)   
    assert cam2img.shape == (3, 3)
    
    cam2img_pad = torch.eye(4) if isinstance(lidar2cam, torch.Tensor) else np.eye(4)
    cam2img_pad[:3, :3] = cam2img
    if isinstance(lidar2cam, np.ndarray):
        lidar2img = np.matmul(cam2img_pad, lidar2cam)
    elif isinstance(lidar2cam, torch.Tensor):
        lidar2img = torch.matmul(cam2img_pad, lidar2cam) 
    else:
        raise TypeError(f'Unsupported data type for lidar2cam as {lidar2cam.dtype}, should be np.ndarray or torch.Tensor') 
    return lidar2img

def pack_lidar2imgs(lidar2cams, cam2imgs):
    """ Pack lidar2imgs from lidar2cams and cam2imgs.
    Args:
        lidar2cams: list, lidar2cam matrix
        cam2imgs: list, cam2img matrix
    Returns:
        lidar2imgs: dict, lidar2img matrix
    """
    lidar2imgs = []
    for view in range(len(lidar2cams)):
        lidar2cam, cam2img = lidar2cams[view], cam2imgs[view]
        lidar2cam = np.array(lidar2cam) if not isinstance(lidar2cam, np.ndarray) else lidar2cam
        cam2img = np.array(cam2img) if not isinstance(cam2img, np.ndarray) else cam2img
        
        lidar2img = get_lidar2img(lidar2cam,  cam2img)
        lidar2imgs.append(lidar2img)
    return lidar2imgs

def tranfrom_worldc_to_camc_light(point_cart, lidar2imgs, ori_size, img_list=None, sem_mask=None):
    """
    Support light-weight projection for point features, without using nusc tokens.
    Transform points from world coordinates to camera coordinates.
    Params:
        voxel_coors: (N, 3) [x, y, z] voxel coordinates or physical point coordinates
    Return:
        points: (N, 3)
    """
    assert len(lidar2imgs) == 6, 'only support nuScenes with 6 views'
    point_projs, masks = [], []
    valid_mask = np.ones_like(point_cart[:, 0]) * -1
    for v in range(len(lidar2imgs)):

        # load pts_semantic_mask for projection check
        point_proj_v, mask = proj_lidar2img(point_cart[:,:3], lidar2imgs[v], 
                                            img_size=ori_size, 
                                            min_dist=1.0, return_original=True)
        ################### save rendered points #######################
        # print('$$$$$$$$$$$$$$$$$$$$ saving rendered points $$$$$$$$$$$$$$$$$$$$$')
        # import cv2
        # color_map = {
        # 0: [0, 0, 0],  # noise                 black
        # 1: [255, 120, 50],  # barrier               orange
        # 2: [255, 192, 203],  # bicycle               pink
        # 3: [255, 255, 0],  # bus                   yellow
        # 4: [0, 150, 245],  # car                   blue
        # 5: [0, 255, 255],  # construction_vehicle  cyan
        # 6: [255, 127, 0],  # motorcycle            dark orange
        # 7: [255, 0, 0],  # pedestrian            red
        # 8: [255, 240, 150],  # traffic_cone          light yellow
        # 9: [135, 60, 0],  # trailer               brown
        # 10: [160, 32, 240],  # truck                 purple
        # 11: [255, 0, 255],  # driveable_surface     dark pink
        # 12: [139, 137, 137],  # other_flat            dark red
        # 13: [75, 0, 75],  # sidewalk              dark purple
        # 14: [150, 240, 80],  # terrain               light green
        # 15: [230, 230, 250],  # manmade               white
        # 16: [0, 175, 0],  # vegetation            green
        # }   
        # VIEW_ORDER = [
        # 'CAM_FRONT',
        # 'CAM_FRONT_RIGHT',
        # 'CAM_FRONT_LEFT',
        # 'CAM_BACK',
        # 'CAM_BACK_LEFT',
        # 'CAM_BACK_RIGHT',
        # ]
        
        # view = VIEW_ORDER[v]
        # sample_token = data_sample.sample_token 
        # vc_sem_mask = sem_mask[mask].detach().cpu().numpy()
        # coloring = np.array([color_map[c] for c in vc_sem_mask])/255
        # im = img_list[bs, v].detach().cpu().numpy().transpose(1, 2, 0)
        # # denormalize and upsample to original size
        
        # # mean = np.array([123.675, 116.28, 103.53]) #TODO-YINING: some prob when denormalizing using this mean and std
        # # std = np.array([58.395, 57.12, 57.375])
        # # denorm_img = (im*255 * std) + mean
        # # min-max normalization
        # denorm_img = (im - im.min()) / (im.max() - im.min()) * 255
        # denorm_img = denorm_img.astype(np.uint8)
        # # upsample
        # im_mmaug = cv2.resize(denorm_img, (ori_size[1], ori_size[0]))
        # import matplotlib.pyplot as plt
        # # plt.imsave(f'output/LaserMix/{sample_token}_{view}_augimg.png', im_mmaug)
        # save_render(voxel_cam_coord.detach().cpu().numpy(), coloring, im_mmaug, dot_size=5, save_path=f'output/rendered_point_PieMix_v2-2_scene/{sample_token}_{view}_sem_mask.png')
        ################################################################################################
        
        point_proj_v[:, [0, 1]] = point_proj_v[:, [1, 0]] # switch (w, h) to (h, w), cause the img feature is (C, H, W)
        point_projs.append(point_proj_v)
        valid_mask[mask] = v
        masks.append(mask)
    
    return point_projs, masks, valid_mask
