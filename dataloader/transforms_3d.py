# Copyright (c) OpenMMLab. All rights reserved.
import random
import re
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose, RandomResize, Resize
from mmdet.datasets.transforms import (PhotoMetricDistortion, RandomCrop,
                                       RandomFlip)
from mmengine import is_list_of, is_tuple_of

from mmdet3d.models.task_modules import VoxelGenerator
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                                LiDARInstance3DBoxes)
from mmdet3d.structures.ops import box_np_ops
from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.datasets.transforms.data_augment_utils import noise_per_object_v3_

# from mmcv.parallel import DataContainer as DC
import h5py
from typing import Any, Dict
from mmdet3d.datasets import GlobalRotScaleTrans

from tools.projection.pc2img import proj_lidar2img
from tools.projection.img_aug import merge_images_yaw, fit_box_cv, expand_box, crop_box_img, paste_box_img, merge_images_pitch_torch, fit_to_box, draw_dashed_box, merge_images_yaw_torch, divide_point_cloud_yaw, divide_point_cloud_pitch, split_yaw, select_mode_by_ratio, update_inst_img
from tools.projection.pc2img import save_render
from tools.projection.instance_insert import InstanceInsertor, sem2inst

@TRANSFORMS.register_module()
class GlobalRotScaleTrans_MM(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""
    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        # if 'transformation_3d_flow' not in input_dict:
        #     input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        # input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']
        lidar_augs = torch.tensor(lidar_augs, dtype=torch.float32)
        
        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = torch.eye(4)
        input_dict['lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']
        # print('********** lidar_aug_matrix: ', input_dict['lidar_aug_matrix'], ' **********')

        return input_dict


@TRANSFORMS.register_module()
class _GlobalRotScaleTransAll(object):
    """Modify: 
    1. rotate along z axis, previous rotate along x axis.
    2. save lidar2cam, lidar2img as torch.Tensor.
    """
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        
        input_dict['points'].translate(trans_factor)
        if 'radar' in input_dict:
            input_dict['radar'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor

        if 'bbox3d_fields' in input_dict:
            for key in input_dict['bbox3d_fields']:
                input_dict[key].translate(trans_factor)

        trans_mat = torch.eye(4)
        trans_factor = torch.tensor(trans_factor)
        trans_mat[:3, -1] = trans_factor
        trans_mat_inv = torch.linalg.inv(trans_mat)
        
        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = torch.eye(4).float()
        input_dict['lidar_aug_matrix'] = trans_mat @ input_dict['lidar_aug_matrix'].float()
        # for view in input_dict["lidar2img"].keys():
        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i] @ trans_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i] @ trans_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ trans_mat_inv
        return

    def _rot_bbox_points(self, input_dict, axis):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if 'rot_degree' not in input_dict:
            rotation = self.rot_range
            noise_rotation = np.random.uniform(rotation[0], rotation[1])
        else:
            noise_rotation = input_dict['rot_degree']

        # if no bbox in input_dict, only rotate points
        assert 'bbox3d_fields' not in input_dict, 'not support rotate bboxes yet'
        if 'rot_degree' not in input_dict:
            rot_mat_T = input_dict['points'].rotate(noise_rotation, axis=axis)
            if 'radar' in input_dict:
                input_dict['radar'].rotate(noise_rotation, axis=axis)
        else:
            rot_mat_T = input_dict['points'].rotate(-noise_rotation, axis=axis)
            if 'radar' in input_dict:
                input_dict['radar'].rotate(-noise_rotation, axis=axis)
        input_dict['pcd_rotation'] = rot_mat_T

        rot_mat = torch.eye(4)
        rot_mat[:3, :3].copy_(rot_mat_T.T)
        # rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0] # rot_mat_T is inversed (transposed) rotate mat
        rot_mat_inv = torch.inverse(rot_mat)
        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = torch.eye(4).float()
        input_dict['lidar_aug_matrix'] = rot_mat @ input_dict['lidar_aug_matrix']

        # for view in input_dict["lidar2img"].keys():
        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i].float() @ rot_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i].float() @ rot_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ rot_mat_inv

        return

        ## not support rotate bboxes yet
        # # rotate points with bboxes
        # for key in input_dict['bbox3d_fields']:
        #     if len(input_dict[key].tensor) != 0:
        #         points, rot_mat_T = input_dict[key].rotate(
        #             noise_rotation, input_dict['points'])
        #         input_dict['points'] = points
        #         input_dict['pcd_rotation'] = rot_mat_T
        #         if 'radar' in input_dict:
        #             input_dict['radar'].rotate(-noise_rotation)

        #         rot_mat = torch.eye(4)
        #         rot_mat[:3, :3].copy_(rot_mat_T)
        #         rot_mat[0, 1], rot_mat[1, 0] = -rot_mat[0, 1], -rot_mat[1, 0] # rot_mat_T is inversed (transposed) rotate mat
        #         rot_mat_inv = torch.inverse(rot_mat)
        #         for view in input_dict["lidar2img"].keys():
        #             input_dict["lidar2img"][view] = torch.tensor(input_dict["lidar2img"][view]).float() @ rot_mat_inv
        #             input_dict["lidar2cam"][view] = torch.tensor(input_dict["lidar2cam"][view]).float() @ rot_mat_inv

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points
        
        if 'radar' in input_dict:
            input_dict['radar'].scale(scale)
            
        if 'bbox3d_fields' in input_dict:
            for key in input_dict['bbox3d_fields']:
                input_dict[key].scale(scale)

        scale_mat = torch.tensor(
            [
                [scale, 0, 0, 0],
                [0, scale, 0, 0],
                [0, 0, scale, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32
        )
        scale_mat_inv = torch.inverse(scale_mat)
        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = torch.eye(4).float()
        input_dict['lidar_aug_matrix'] = scale_mat @ input_dict['lidar_aug_matrix'].float()

        # for view in input_dict["lidar2img"].keys():
        for i in range(len(input_dict["lidar2img"])):
            if isinstance(input_dict["lidar2img"][i], np.ndarray):
                input_dict["lidar2img"][i] = torch.tensor(input_dict["lidar2img"][i])
                input_dict["lidar2cam"][i] = torch.tensor(input_dict["lidar2cam"][i])
            input_dict["lidar2img"][i] = input_dict["lidar2img"][i].float() @ scale_mat_inv
            input_dict["lidar2cam"][i] = input_dict["lidar2cam"][i].float() @ scale_mat_inv
        if "lidars2imgs_mix_torch" in input_dict:
            for i in range(len(input_dict["lidars2imgs_mix_torch"])):
                if isinstance(input_dict["lidars2imgs_mix_torch"][i], np.ndarray):
                    input_dict["lidars2imgs_mix_torch"][i] = torch.tensor(input_dict["lidars2imgs_mix_torch"][i], dtype=input_dict["lidar2img"][i].dtype)
                input_dict["lidars2imgs_mix_torch"][i] = input_dict["lidars2imgs_mix_torch"][i] @ scale_mat_inv
        return
    
    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict, rot_axis=2):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            rot_axis (int): Rotation axis. Defaults to z-axis.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict, axis=rot_axis)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)
                
        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip3D_MM:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = torch.eye(3)
        if flip_horizontal:
            rotation = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = torch.eye(4).float()
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        
        rotation_pad = torch.eye(4)
        rotation_pad[:3, :3] = rotation
        rotation_inv_pad = torch.inverse(rotation_pad)
        for i in range(len(data["lidar2img"])):
            if isinstance(data["lidar2img"][i], np.ndarray):
                data["lidar2img"][i] = torch.tensor(data["lidar2img"][i])
                data["lidar2cam"][i] = torch.tensor(data["lidar2cam"][i])
            data["lidar2img"][i] = data["lidar2img"][i].float() @ rotation_inv_pad
            data["lidar2cam"][i] = data["lidar2cam"][i].float() @ rotation_inv_pad
        if "lidars2imgs_mix_torch" in data:
            for i in range(len(data["lidars2imgs_mix_torch"])):
                if isinstance(data["lidars2imgs_mix_torch"][i], np.ndarray):
                    data["lidars2imgs_mix_torch"][i] = torch.tensor(data["lidars2imgs_mix_torch"][i], dtype=data["lidar2img"][i].dtype)
                data["lidars2imgs_mix_torch"][i] = data["lidars2imgs_mix_torch"][i] @ rotation_inv_pad
        return data


@TRANSFORMS.register_module(force=True)
class _PolarMix_MM(BaseTransform):
    """PolarMix data augmentation.

    The polarmix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Exchange sectors of two point clouds that are cut with certain
           azimuth angles.
        3. Cut point instances from picked point cloud, rotate them by multiple
           azimuth angles, and paste the cut and rotated instances.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        instance_classes (List[int]): Semantic masks which represent the
            instance.
        swap_ratio (float): Swap ratio of two point cloud. Defaults to 0.5.
        rotate_paste_ratio (float): Rotate paste ratio. Defaults to 1.0.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def polar_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        # save binary mask 
        binary_mask = torch.ones(len(points))
        binary_mask_orig = torch.ones(len(points))
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_panoptic = True

        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            #############################
            # check whether use PolarMix
            # print('********************* USING PolarMix_Swap NOW *********************')
            #############################
        
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:,
                                                                            0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            a, b = points[idx].shape[0], mix_points[mix_idx].shape[0]
            swap_points = points.cat([points[idx].clone(), mix_points[mix_idx]])
            swap_pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)
            
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                    (pts_instance_mask[idx.numpy()],
                    mix_instance_mask[mix_idx.numpy()]),
                    axis=0)        
                        
            # update points
            points = swap_points
            pts_semantic_mask = swap_pts_semantic_mask
            # save binary mask
            binary_mask = torch.ones(len(points))
            binary_mask[a:a+b] = 0
            # save binary mask for original points
            binary_mask_orig[a:] = 0

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points, instance_pts_semantic_mask = [], []
            if mix_panoptic:
                instance_pts_instance_mask = []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
                if mix_panoptic:
                    instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
                    
            
            instance_points = mix_points.cat(instance_points)
            # print(instance_points.shape)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)
            if mix_panoptic:
               instance_pts_instance_mask = np.concatenate(
                instance_pts_instance_mask, axis=0) 

            # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            if mix_panoptic:
                copy_pts_instance_mask = [instance_pts_instance_mask]
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            for angle in angle_list:
                new_points = instance_points.clone()
                new_points.rotate(angle)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
                if mix_panoptic:
                    copy_pts_instance_mask.append(instance_pts_instance_mask)
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)
            if mix_panoptic:
                copy_pts_instance_mask = np.concatenate(
                copy_pts_instance_mask, axis=0)

            c = copy_points.shape[0]
            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                (pts_instance_mask, copy_pts_instance_mask), axis=0)
            
            # save binary mask
            binary_mask = torch.cat([binary_mask, torch.zeros(c)])  # Mark rotated-pasted points as modified
            # save binary mask for original points
            binary_mask_orig = binary_mask_orig # pasting does not change the original points
            
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
            
        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig  
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module(force=True)
class _PolarMix_MM_IMG(BaseTransform):
    """PolarMix data augmentation.

    The polarmix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Exchange sectors of two point clouds that are cut with certain
           azimuth angles.
        3. Cut point instances from picked point cloud, rotate them by multiple
           azimuth angles, and paste the cut and rotated instances.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)
    - images (dict): meta data of images: {'VIEW_NAME': {'img_path': str, 'lidar2img': np.ndarray}}
    - img (list): list of view images in np.ndarray
    
    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - img (list): list of view images in np.ndarray
    
    Args:
        instance_classes (List[int]): Semantic masks which represent the
            instance.
        swap_ratio (float): Swap ratio of two point cloud. Defaults to 0.5.
        rotate_paste_ratio (float): Rotate paste ratio. Defaults to 1.0.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 img_aug: bool,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def polar_mix_transform(self, input_dict: dict, mix_results: dict, angle=None) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        points = input_dict['points']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        block_cp = []
        block_cp_instlabel, block_cp_semlabel = [], []
        block_ro, block_ro_instlabel, block_ro_semlabel = [], [], []
        out_im_list = []
        binary_mask = torch.ones(len(points))
        binary_mask_orig = torch.ones(len(points))
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_panoptic = True
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            assert sam_pmask_2d_mix.dtype == np.int64
            sam_pmask_2d_mix[sam_pmask_2d_mix>0] += (1000<<16) # not overlap id
            sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            # out_sam_pmask_2d, out_sam_pscore_2d = [], []
            mix_sam_pred = True
            # print('ids in sam_pmask_2d_mix', np.unique(sam_pmask_2d_mix))
            # print('ids in sam_pmask_2d', np.unique(sam_pmask_2d))
            
        mix_tgt_pl = False
        if 'UDA' in mix_results:
            pl_psem = input_dict['UDA']['tgt_pl_psem']
            pl_pins = input_dict['UDA']['tgt_pl_pins']
            pl_pscore = input_dict['UDA']['tgt_pl_pscore']
            pl_psem_mix = mix_results['UDA']['tgt_pl_psem']
            pl_pins_mix = mix_results['UDA']['tgt_pl_pins']
            pl_pscore_mix = mix_results['UDA']['tgt_pl_pscore']
            mix_tgt_pl = True
         
        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            # if angle is not None: 
            #     start_angle = angle
            # else:
            # print(f'********start angle {start_angle}, {start_angle/np.pi*180}')

            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:,
                                                                            0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            a, b = points[idx].shape[0], mix_points[mix_idx].shape[0]
            block_orig = points[idx].coord[:, :3]
            block_swap = mix_points[mix_idx].coord[:, :3]
            block_sem_orig = pts_semantic_mask[idx.numpy()]
            block_sem_swap = mix_pts_semantic_mask[mix_idx.numpy()]
            
            swap_points = points.cat([points[idx], mix_points[mix_idx]])
            swap_pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)
            
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                    (pts_instance_mask[idx.numpy()],
                    mix_instance_mask[mix_idx.numpy()]),
                    axis=0)  
                
            if mix_sam_pred:  
                sam_pmask_2d = np.concatenate(
                    (sam_pmask_2d[idx.numpy()],
                    sam_pmask_2d_mix[mix_idx.numpy()]),
                    axis=0)
                sam_pscore_2d = np.concatenate(
                    (sam_pscore_2d[idx.numpy()],
                    sam_pscore_2d_mix[mix_idx.numpy()]),
                    axis=0)    
                
            if mix_tgt_pl:
                pl_psem = np.concatenate(
                    (pl_psem[idx.numpy()],
                    pl_psem_mix[mix_idx.numpy()]),
                    axis=0)
                pl_pins = np.concatenate(
                    (pl_pins[idx.numpy()],
                    pl_pins_mix[mix_idx.numpy()]),
                    axis=0)
                pl_pscore = np.concatenate(
                    (pl_pscore[idx.numpy()],
                    pl_pscore_mix[mix_idx.numpy()]),
                    axis=0)
            
            # print(f'2********point (mixed) shape {swap_points.shape}, binary mask shape {binary_mask.shape}, a {a}, b {b}')
            
            # update points and binary mask
            points = swap_points
            pts_semantic_mask = swap_pts_semantic_mask
            binary_mask = torch.ones(len(points))
            binary_mask[a:a+b] = 0

            ########## image augmentation for yaw swap ##########
            if self.img_aug:
                image_list_orig = input_dict['img']
                image_list_mix = mix_results['img']
                image_list_orig = [torch.tensor(img) for img in image_list_orig]
                image_list_mix  = [torch.tensor(img) for img in image_list_mix]
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                
                img_size = input_dict['img_shape'][:2] # (360, 630)
                ori_size = input_dict['ori_shape'] # (900, 1600)
                img_scale = input_dict['scale_factor'][0] # 0.4
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                # if isinstance(lidar2img_orig, dict):
                    # lidar2img_orig = list(lidar2img_orig.values())
                    # lidar2img_mix  = list(lidar2img_mix.values())
                
                for v in range(len(image_list_orig)):
                    im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                    block_points_img, mask = proj_lidar2img(block_orig, lidar2img_orig[v], 
                                                            img_size=(ori_size[1], ori_size[0]), 
                                                            min_dist=1.0)
                    block_points_img = block_points_img * img_scale
                    l, r = fit_to_box(block_points_img.numpy(), 'vertical')

                    img_new = merge_images_yaw(im_orig, im_mix, l, r)
                    ########## save image for debug ##########
                    # sample_token = input_dict['token']
                    # view = list(input_dict['images'].keys())[v]
                    # denorm_img = (img_new - img_new.min()) / (img_new.max() - img_new.min()) * 255
                    # denorm_img = denorm_img.astype(np.uint8)
                    # cv2.imwrite(f'misc/rendered_point_mmaug/{sample_token}_{view}_polar_swap.png', denorm_img)
                    ########## save image for debug ##########
                    out_im_list.append(img_new)
                input_dict['img'] = out_im_list
            ########## image augmentation ##########
            
        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points, instance_pts_semantic_mask = [], []
            if mix_panoptic:
                instance_pts_instance_mask = []
            if mix_sam_pred:
                instance_pts_sam_pmask_2d, instance_pts_sam_pscore_2d = [], []
            if mix_tgt_pl:
                instance_pl_psem, instance_pl_pins, instance_pl_pscore = [], [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
                if mix_panoptic:
                    instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
                if mix_sam_pred:
                    instance_pts_sam_pmask_2d.append(sam_pmask_2d_mix[mix_idx])
                    instance_pts_sam_pscore_2d.append(sam_pscore_2d_mix[mix_idx])
                if mix_tgt_pl:
                    instance_pl_psem.append(pl_psem_mix[mix_idx])
                    instance_pl_pins.append(pl_pins_mix[mix_idx])
                    instance_pl_pscore.append(pl_pscore_mix[mix_idx])
                    
            block_cp = instance_points.copy()
            block_cp_instlabel = instance_pts_instance_mask.copy()
            block_cp_semlabel  = instance_pts_semantic_mask.copy()
            # print(f'4******** add instance points, number of inst {np.unique(instance_pts_semantic_mask).shape}')
            
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)
            if mix_panoptic:
               instance_pts_instance_mask = np.concatenate(
                instance_pts_instance_mask, axis=0)
            if mix_sam_pred:
                instance_pts_sam_pmask_2d = np.concatenate(
                    instance_pts_sam_pmask_2d, axis=0)
                instance_pts_sam_pscore_2d = np.concatenate(
                    instance_pts_sam_pscore_2d, axis=0) 
            # bc = instance_points.shape[0]
            if mix_tgt_pl:
                instance_pl_psem = np.concatenate(instance_pl_psem, axis=0)
                instance_pl_pins = np.concatenate(instance_pl_pins, axis=0)
                instance_pl_pscore = np.concatenate(instance_pl_pscore, axis=0)
                
            # # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            if mix_panoptic:
                copy_pts_instance_mask = [instance_pts_instance_mask]
            if mix_sam_pred:
                copy_pts_sam_pmask_2d = [instance_pts_sam_pmask_2d]
                copy_pts_sam_pscore_2d = [instance_pts_sam_pscore_2d]
            if mix_tgt_pl:
                copy_pl_psem = [instance_pl_psem]
                copy_pl_pins = [instance_pl_pins]
                copy_pl_pscore = [instance_pl_pscore]
                
            angle_list = [
                # angle * np.pi,
                # 0.5 * np.pi * 2 / 3,
                # np.random.random() * np.pi * 2 / 3,
                # (np.random.random() + 1) * np.pi * 2 / 3
            ]
            # print(f'$$$$$$$$$$$$$$$$ Warning: instance rotate has not implemented yet!!!! ')
            for angle in angle_list:
                # print(f'instance rotate angle at: {angle}, {angle/np.pi*180}')
                new_points = instance_points.clone()
                # new_points.rotate(angle, axis=0)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
                if mix_panoptic:
                    copy_pts_instance_mask.append(instance_pts_instance_mask)
                if mix_sam_pred:
                    copy_pts_sam_pmask_2d.append(instance_pts_sam_pmask_2d)
                    copy_pts_sam_pscore_2d.append(instance_pts_sam_pscore_2d)
                if mix_tgt_pl:
                    copy_pl_psem.append(instance_pl_psem)
                    copy_pl_pins.append(instance_pl_pins)
                    copy_pl_pscore.append(instance_pl_pscore)
                block_ro.append(new_points)
                block_ro_instlabel.append(instance_pts_instance_mask.copy())
                block_ro_semlabel.append(instance_pts_semantic_mask.copy())
                
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)
            if mix_panoptic:
                copy_pts_instance_mask = np.concatenate(
                copy_pts_instance_mask, axis=0)
            if mix_sam_pred:
                copy_pts_sam_pmask_2d = np.concatenate(
                    copy_pts_sam_pmask_2d, axis=0)
                copy_pts_sam_pscore_2d = np.concatenate(
                    copy_pts_sam_pscore_2d, axis=0)
            if mix_tgt_pl:
                copy_pl_psem = np.concatenate(copy_pl_psem, axis=0)
                copy_pl_pins = np.concatenate(copy_pl_pins, axis=0)
                copy_pl_pscore = np.concatenate(copy_pl_pscore, axis=0)
            c = copy_points.shape[0]
            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                (pts_instance_mask, copy_pts_instance_mask), axis=0)
            if mix_sam_pred:
                sam_pmask_2d = np.concatenate(
                    (sam_pmask_2d, copy_pts_sam_pmask_2d), axis=0)
                sam_pscore_2d = np.concatenate(
                    (sam_pscore_2d, copy_pts_sam_pscore_2d), axis=0)
            if mix_tgt_pl:
                pl_psem = np.concatenate((pl_psem, copy_pl_psem), axis=0)
                pl_pins = np.concatenate((pl_pins, copy_pl_pins), axis=0)
                pl_pscore = np.concatenate((pl_pscore, copy_pl_pscore), axis=0)
            # save binary mask
            binary_mask = torch.cat([binary_mask, torch.zeros(c)])  # Mark rotated-pasted points as modified
            binary_mask_orig = binary_mask_orig # pasting does not change the original points
            # print(f'3********point shape {points.shape}, added point shape {c}, binary mask shape {binary_mask.shape}')

            ########## image augmentation for instance copy ##########
            if self.img_aug:
                if len(out_im_list) == 0: # if not swapped
                    out_im_list = input_dict['img']
                
                # TODO-YINING: pack pre-processings into a function
                image_list_orig = out_im_list
                image_list_mix = mix_results['img']
                image_list_orig = [torch.tensor(img) for img in image_list_orig]
                image_list_mix  = [torch.tensor(img) for img in image_list_mix]
                
                img_size = input_dict['img_shape'][:2] # (360, 630)
                ori_size = input_dict['ori_shape'] # (900, 1600)
                img_scale = input_dict['scale_factor'][0] # 0.4
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                # if isinstance(lidar2img_orig, dict):
                #     lidar2img_orig = list(lidar2img_orig.values())
                #     lidar2img_mix  = list(lidar2img_mix.values())
                
                cp_coord = torch.cat([inst.coord for inst in block_cp])
                cp_instlabel = np.concatenate(block_cp_instlabel) # at least know the number of cp
                cp_semlabel  = np.concatenate(block_cp_semlabel)
                N = binary_mask.shape[0]
                N_cp = cp_instlabel.shape[0]
                N_ro = 0
                
                if N_cp > 0:
                    # cp_mask = np.zeros(binary_mask.shape).astype(np.bool_)
                    # cp_mask[N-(N_cp+N_ro):N-N_ro] = 1
                    
                    for i in range(len(image_list_orig)):
                        im_orig = image_list_orig[i].numpy()
                        im_1_copy = im_orig.copy()
                        im_mix = image_list_mix[i].numpy()
                        # print(f'$$$$$$$$$$$$$$$$ Warning: here should be lidar2img_mix[i] instead of lidar2img_orig[i] ')
                        cp_coord_img, mask = proj_lidar2img(cp_coord, lidar2img_mix[i], 
                                                            img_size=(ori_size[1], ori_size[0]), 
                                                            min_dist=1.0)
                        cp_coord_img = cp_coord_img * img_scale

                        cp_coord_img = cp_coord_img.numpy()
                        # print('$'*20)
                        # print(cp_instlabel)
                        # print(mask)
                        # print('$'*20)
                        
                        cp_instlabel_v = cp_instlabel[mask.detach().cpu().numpy()]
                        cp_semlabel_v  = cp_semlabel[mask.detach().cpu().numpy()]
                        # coloring = np.array([color_map[k] for k in cp_semlabel_v])/255
                        
                        for inst_id in np.unique(cp_instlabel_v):
                            inst_coord = cp_coord_img[cp_instlabel_v == inst_id]
                            inst_cls = cp_semlabel_v[cp_instlabel_v == inst_id][0]
                            # inst_coord = inst_coord[:2, :].T
                            box = fit_box_cv(inst_coord)
                            ##1.  expand the box
                            # box = expand_box_proportional(box, 0.2, im_2_np.shape[:2])
                            box_ex = expand_box(box, 10, im_mix.shape[:2])
                            ##2.  crop
                            crop_img = crop_box_img(box_ex, im_mix)
                            ##3.paste to orig image
                            im_1_copy = paste_box_img(box_ex, im_orig, crop_img)
                            ########## save image for debug ##########
                            ##3.  draw the box
                            # im_save = im_1_copy.copy()
                            # im_save = draw_dashed_box(im_save, box_ex, thickness=10, dash_length=5)
                            # sample_token = input_dict['token']
                            # view = list(input_dict['images'].keys())[i]
                            # denorm_img = (im_save - im_save.min()) / (im_save.max() - im_save.min()) * 255
                            # denorm_img = denorm_img.astype(np.uint8)
                            # cv2.imwrite(f'misc/rendered_point_mmaug/{sample_token}_{view}_polar_inst.png', denorm_img)
                            ########## save image for debug ##########

                        # plot_point_in_camview_2(cp_coord_img, coloring, im_1_copy, dot_size=3)
                        out_im_list[i] = im_1_copy
            ########## image augmentation ##########
        
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            assert sam_pmask_2d.shape[0] == pts_semantic_mask.shape[0]
            assert sam_pscore_2d.shape[0] == pts_semantic_mask.shape[0]
            input_dict['sam_pmask_2d'] = sam_pmask_2d
            input_dict['sam_pscore_2d'] = sam_pscore_2d
        if mix_tgt_pl:
            assert pl_psem.shape[0] == pts_semantic_mask.shape[0]
            input_dict['UDA']['tgt_pl_psem'] = pl_psem
            input_dict['UDA']['tgt_pl_pins'] = pl_pins
            input_dict['UDA']['tgt_pl_pscore'] = pl_pscore
            # print('ids in augmented sam_pmask_2d', np.unique(sam_pmask_2d))
        # input_dict['binary_mask'] = binary_mask
        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig  
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        
        # input_dict['block_orig'] = block_orig
        # input_dict['block_swap'] = block_swap
        # input_dict['block_cp'] = block_cp
        # input_dict['block_cp_instlabel'] = block_cp_instlabel
        # input_dict['block_cp_semlabel'] = block_cp_semlabel
        # input_dict['block_ro'] = block_ro
        # input_dict['block_ro_instlabel'] = block_ro_instlabel
        # input_dict['block_ro_semlabel'] = block_ro_semlabel
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

# V1: for coupled vertical cut&swap, instance copy&rotate.
@TRANSFORMS.register_module(force=True)
class _PieMix(BaseTransform):
    """PierMix data augmentation. 
    """
    def __init__(self,
                 img_aug: bool,
                 num_areas: List[int],
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.img_aug = img_aug
        self.num_areas = num_areas
        
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio
        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def pie_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """
        split the point cloud into several regions according to yaw angles and combine the areas crossly.
        """
        points = input_dict['points']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        binary_mask = None
        binary_mask_orig = torch.zeros(len(points))
        
        
        # for instance copy and rotate
        block_cp = []
        block_cp_instlabel, block_cp_semlabel = [], []
        block_ro, block_ro_instlabel, block_ro_semlabel = [], [], []
        out_im_list = []
        
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            mix_panoptic = True
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            assert sam_pmask_2d_mix.dtype == np.int64
            sam_pmask_2d_mix[sam_pmask_2d_mix>0] += (1000<<16) # not overlap id
            sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            out_sam_pmask_2d, out_sam_pscore_2d = [], []
            mix_sam_pred = True
            
        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            num_areas = np.random.choice(self.num_areas, size=1)[0]
            area_splits = divide_point_cloud_yaw(num_areas)
            
            out_points = []
            out_pts_semantic_mask, out_pts_instance_mask = [], []
            block_orig, block_mix = [], []
            
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:, 0])

            for i, area_split in enumerate(area_splits):
                # convert angle to radian
                start_angle = area_split[0]
                end_angle = area_split[1]
                if i % 2 == 0:  # pick from original point cloud
                    idx = split_yaw(yaw, start_angle, end_angle)
                    selected_points = points[idx]
                    
                    out_points.append(points[idx])
                    block_orig.append(points[idx])
                    out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                    if mix_panoptic:
                        out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
                    if mix_sam_pred:
                        out_sam_pmask_2d.append(sam_pmask_2d[idx.numpy()])
                        out_sam_pscore_2d.append(sam_pscore_2d[idx.numpy()])
                    if binary_mask == None:
                        binary_mask = torch.ones(selected_points.shape[0], dtype=torch.uint8)
                    else:
                        binary_mask = torch.cat([binary_mask, torch.ones(selected_points.shape[0], dtype=torch.uint8)])  # Mark as original
                    binary_mask_orig[idx] = 1  
                    
                else:  # pickle from mixed point cloud
                    idx = split_yaw(mix_yaw, start_angle, end_angle)
                    selected_points = mix_points[idx]
                    
                    out_points.append(mix_points[idx])
                    block_mix.append(mix_points[idx])
                    out_pts_semantic_mask.append(
                        mix_pts_semantic_mask[idx.numpy()])

                    if mix_panoptic:
                        out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
                    if mix_sam_pred:
                        out_sam_pmask_2d.append(sam_pmask_2d_mix[idx.numpy()])
                        out_sam_pscore_2d.append(sam_pscore_2d_mix[idx.numpy()])
                    binary_mask = torch.cat([binary_mask, torch.zeros(selected_points.shape[0], dtype=torch.uint8)])  # Mark as modified
            points = points.cat(out_points)
            pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
            if mix_sam_pred:
                sam_pmask_2d = np.concatenate(out_sam_pmask_2d, axis=0)
                sam_pscore_2d = np.concatenate(out_sam_pscore_2d, axis=0)
                
            
            ########## image augmentation for yaw swap ##########
            if self.img_aug:
                out_im_list = []
                image_list_orig = input_dict['img']
                image_list_mix = mix_results['img']
                image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
                image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                
                img_size = input_dict['img_shape'][:2] # (360, 630)
                ori_size = input_dict['ori_shape'] # (900, 1600)
                img_scale = input_dict['scale_factor'][0] # 0.4
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                # if isinstance(lidar2img_orig, dict):
                    # lidar2img_orig = list(lidar2img_orig.values())
                    # lidar2img_mix  = list(lidar2img_mix.values())
                
                for v in range(len(image_list_orig)):
                    im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                    lidar2img_mix_v = lidar2img_mix[v] # @ torch.inverse(aug_mat)
                    for i in range(len(block_mix)):
                        pblock = block_mix[i].coord
                        # sem_mask_b = new_sem_mask[i]
                        points_img_b, mask_b = proj_lidar2img(pblock, lidar2img_mix_v, 
                                                            img_size=(ori_size[1], ori_size[0]),  # w, h
                                                            min_dist=1.0)
                        points_img_b = points_img_b*img_scale
                        mixed_img, block_mask = merge_images_yaw_torch(im_orig, im_mix, points_img_b)   
                        im_orig = mixed_img
                        
                        

                    # plt.imshow(im_orig[:,:,[2,1,0]])
                    # plt.show()
                    out_im_list.append(im_orig)
                    
                    ########## save image for debug ##########
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
                    # # sample_token = data_sample.sample_token 
                    # vc_sem_mask = ou[mask_b]
                    # coloring = np.array([color_map[c] for c in vc_sem_mask])/255
                    # im = mixed_img.detach().cpu().numpy()
                    # # denormalize and upsample to original size
                    
                    # # mean = np.array([123.675, 116.28, 103.53]) #TODO-YINING: some prob when denormalizing using this mean and std
                    # # std = np.array([58.395, 57.12, 57.375])
                    # # denorm_img = (im*255 * std) + mean
                    # # min-max normalization
                    # denorm_img = (im - im.min()) / (im.max() - im.min()) * 255
                    # denorm_img = denorm_img.astype(np.uint8)
                    # # upsample
                    # # im_mmaug = cv2.resize(denorm_img, (ori_size[1], ori_size[0]))
                    # # import matplotlib.pyplot as plt
                    # # plt.imsave(f'output/piemix/{sample_token}_{view}_augimg.png', im_mmaug)
                    # import time
                    # name = time.time()
                    # plt.savefig(f'output/rendered_point_piemix_before/{view}_{name}_img.png')
                    # save_render(points_img_b.detach().cpu().numpy(), coloring, denorm_img, dot_size=5, save_path=f'output/rendered_point_piemix_before/{view}_{name}_sem_mask.png')
                    ########## save image for debug ##########
                # input_dict['img'] = out_im_list
            ########## image augmentation ##########
        else:
            binary_mask = torch.ones(len(points))
            binary_mask_orig = torch.ones(len(points)) 
            
        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            # N = pts_semantic_mask.shape[0] # shape of points before instance augmentation
            instance_points, instance_pts_semantic_mask = [], []
            if mix_panoptic:
                instance_pts_instance_mask = []
            if mix_sam_pred:
                instance_pts_sam_pmask_2d, instance_pts_sam_pscore_2d = [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
                if mix_panoptic:
                    instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
                if mix_sam_pred:
                    instance_pts_sam_pmask_2d.append(sam_pmask_2d_mix[mix_idx])
                    instance_pts_sam_pscore_2d.append(sam_pscore_2d_mix[mix_idx])
            
            block_cp = instance_points.copy()
            block_cp_instlabel = instance_pts_instance_mask.copy()
            block_cp_semlabel  = instance_pts_semantic_mask.copy()
            # print(f'4******** add instance points, number of inst {np.unique(instance_pts_semantic_mask).shape}')
            
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)
            if mix_panoptic:
               instance_pts_instance_mask = np.concatenate(
                instance_pts_instance_mask, axis=0)
            if mix_sam_pred:
                instance_pts_sam_pmask_2d = np.concatenate(
                    instance_pts_sam_pmask_2d, axis=0)
                instance_pts_sam_pscore_2d = np.concatenate(
                    instance_pts_sam_pscore_2d, axis=0) 
            # bc = instance_points.shape[0]

            # # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            if mix_panoptic:
                copy_pts_instance_mask = [instance_pts_instance_mask]
            if mix_sam_pred:
                copy_pts_sam_pmask_2d = [instance_pts_sam_pmask_2d]
                copy_pts_sam_pscore_2d = [instance_pts_sam_pscore_2d]
            angle_list = [
                # angle * np.pi,
                # 0.5 * np.pi * 2 / 3,
                # np.random.random() * np.pi * 2 / 3,
                # (np.random.random() + 1) * np.pi * 2 / 3
            ]
            # print(f'$$$$$$$$$$$$$$$$ Warning: instance rotate has not implemented yet!!!! ')
            for angle in angle_list:
                print(f'instance rotate angle at: {angle}, {angle/np.pi*180}')
                new_points = instance_points.clone()
                # new_points.rotate(angle, axis=0)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
                if mix_panoptic:
                    copy_pts_instance_mask.append(instance_pts_instance_mask)
                if mix_sam_pred:
                    copy_pts_sam_pmask_2d.append(instance_pts_sam_pmask_2d)
                    copy_pts_sam_pscore_2d.append(instance_pts_sam_pscore_2d)
                block_ro.append(new_points)
                block_ro_instlabel.append(instance_pts_instance_mask.copy())
                block_ro_semlabel.append(instance_pts_semantic_mask.copy())
                
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)
            if mix_panoptic:
                copy_pts_instance_mask = np.concatenate(
                copy_pts_instance_mask, axis=0)
            if mix_sam_pred:
                copy_pts_sam_pmask_2d = np.concatenate(
                    copy_pts_sam_pmask_2d, axis=0)
                copy_pts_sam_pscore_2d = np.concatenate(
                    copy_pts_sam_pscore_2d, axis=0)

            c = copy_points.shape[0]
            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)
            if mix_panoptic:
                pts_instance_mask = np.concatenate(
                (pts_instance_mask, copy_pts_instance_mask), axis=0)
            if mix_sam_pred:
                sam_pmask_2d = np.concatenate(
                    (sam_pmask_2d, copy_pts_sam_pmask_2d), axis=0)
                sam_pscore_2d = np.concatenate(
                    (sam_pscore_2d, copy_pts_sam_pscore_2d), axis=0)
            
            # save binary mask
            # if binary_mask == None:
            #     binary_mask = torch.ones(N, dtype=torch.uint8)
            binary_mask = torch.cat([binary_mask, torch.zeros(c)])  # Mark rotated-pasted points as modified
            binary_mask_orig = binary_mask_orig # pasting does not change the original points
            # print(f'3********point shape {points.shape}, added point shape {c}, binary mask shape {binary_mask.shape}')

            ########## image augmentation for instance copy ##########
            if self.img_aug:
                if len(out_im_list) == 0: # if not swapped
                    out_im_list = input_dict['img']
                
                # TODO-YINING: pack pre-processings into a function
                image_list_orig = out_im_list
                image_list_mix = mix_results['img']
                image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
                image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
                
                img_size = input_dict['img_shape'][:2] # (360, 630)
                ori_size = input_dict['ori_shape'] # (900, 1600)
                img_scale = input_dict['scale_factor'][0] # 0.4
                assert int(img_size[0]) == int(ori_size[0] * img_scale)
                
                lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
                # if isinstance(lidar2img_orig, dict):
                #     lidar2img_orig = list(lidar2img_orig.values())
                #     lidar2img_mix  = list(lidar2img_mix.values())
                
                cp_coord = torch.cat([inst.coord for inst in block_cp])
                cp_instlabel = np.concatenate(block_cp_instlabel) # at least know the number of cp
                cp_semlabel  = np.concatenate(block_cp_semlabel)
                N = binary_mask.shape[0]
                N_cp = cp_instlabel.shape[0]
                N_ro = 0
                
                if N_cp > 0:
                    # cp_mask = np.zeros(binary_mask.shape).astype(np.bool_)
                    # cp_mask[N-(N_cp+N_ro):N-N_ro] = 1
                    
                    for i in range(len(image_list_orig)):
                        im_orig = image_list_orig[i].numpy()
                        im_1_copy = im_orig.copy()
                        im_mix = image_list_mix[i].numpy()
                        cp_coord_img, mask = proj_lidar2img(cp_coord, lidar2img_mix[i], 
                                                            img_size=(ori_size[1], ori_size[0]), 
                                                            min_dist=1.0)
                        cp_coord_img = cp_coord_img * img_scale

                        cp_coord_img = cp_coord_img.numpy()
                        # print('$'*20)
                        # print(cp_instlabel)
                        # print(mask)
                        # print('$'*20)
                        
                        cp_instlabel_v = cp_instlabel[mask.detach().cpu().numpy()]
                        cp_semlabel_v  = cp_semlabel[mask.detach().cpu().numpy()]
                        # coloring = np.array([color_map[k] for k in cp_semlabel_v])/255
                        
                        for inst_id in np.unique(cp_instlabel_v):
                            inst_coord = cp_coord_img[cp_instlabel_v == inst_id]
                            inst_cls = cp_semlabel_v[cp_instlabel_v == inst_id][0]
                            # inst_coord = inst_coord[:2, :].T
                            box = fit_box_cv(inst_coord)
                            ##1.  expand the box
                            # box = expand_box_proportional(box, 0.2, im_2_np.shape[:2])
                            box_ex = expand_box(box, 10, im_mix.shape[:2])
                            ##2.  crop
                            crop_img = crop_box_img(box_ex, im_mix)
                            ##3.paste to orig image
                            im_1_copy = paste_box_img(box_ex, im_orig, crop_img)
                            ########## save image for debug ##########
                            ##3.  draw the box
                            # im_save = im_1_copy.copy()
                            # im_save = draw_dashed_box(im_save, box_ex, thickness=10, dash_length=5)
                            # sample_token = input_dict['token']
                            # view = list(input_dict['images'].keys())[i]
                            # denorm_img = (im_save - im_save.min()) / (im_save.max() - im_save.min()) * 255
                            # denorm_img = denorm_img.astype(np.uint8)
                            # cv2.imwrite(f'misc/rendered_point_mmaug/{sample_token}_{view}_polar_inst.png', denorm_img)
                            ########## save image for debug ##########

                        # plot_point_in_camview_2(cp_coord_img, coloring, im_1_copy, dot_size=3)
                        out_im_list[i] = im_1_copy
            ########## image augmentation ##########
        
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            assert sam_pmask_2d.shape[0] == pts_semantic_mask.shape[0]
            assert sam_pscore_2d.shape[0] == pts_semantic_mask.shape[0]
            input_dict['sam_pmask_2d'] = sam_pmask_2d
            input_dict['sam_pscore_2d'] = sam_pscore_2d
        
        assert binary_mask.shape[0] == points.coord.shape[0]
        # print(torch.count_nonzero(binary_mask), torch.count_nonzero(binary_mask_orig))
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig  
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        
        # input_dict['block_orig'] = block_orig
        # input_dict['block_swap'] = block_swap
        # input_dict['block_cp'] = block_cp
        # input_dict['block_cp_instlabel'] = block_cp_instlabel
        # input_dict['block_cp_semlabel'] = block_cp_semlabel
        # input_dict['block_ro'] = block_ro
        # input_dict['block_ro_instlabel'] = block_ro_instlabel
        # input_dict['block_ro_semlabel'] = block_ro_semlabel
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.pie_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'yaw_angles={self.yaw_angles}, '
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

# V2: for decoupled horizontal/vertical cut&swap.
@TRANSFORMS.register_module(force=True)
class _PieMix_Scene(BaseTransform):
    """PierMix data augmentation. 
    """
    def __init__(self,
                 img_aug: bool,
                 hv_ratios: List[int],
                 h_num_areas: List[int],
                 v_num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        
        assert is_list_of(h_num_areas, int), 'num_areas should be a list of int.'
        assert is_list_of(v_num_areas, int), 'num_areas should be a list of int.'
        self.h_num_areas = h_num_areas
        self.v_num_areas = v_num_areas
        self.mode = select_mode_by_ratio(hv_ratios)
        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles
        if self.pitch_angles[1] > np.pi: # convert to radian
            self.pitch_angles[0] = self.pitch_angles[0]/180*np.pi
            self.pitch_angles[1] = self.pitch_angles[1]/180*np.pi
            
        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
            
    def pie_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """
        split the point cloud into several regions according to yaw angles and combine the areas crossly.
        """
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            pts_instance_mask = input_dict['pts_instance_mask']
            _, pts_instance_mask = np.unique(pts_instance_mask, return_inverse=True) # re-index
            N = pts_instance_mask.max()
            mix_instance_mask = mix_results['pts_instance_mask']
            _, mix_instance_mask = np.unique(mix_instance_mask, return_inverse=True) # re-index
            # print(f'orig instance num: {N}, new instance num: {mix_instance_mask.max()}')
            mix_panoptic = True
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            out_sam_pmask_2d, out_sam_pscore_2d = [], []
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            assert sam_pmask_2d.dtype == np.int64
            N_sam = sam_pmask_2d.max()
            sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            _, sam_pmask_2d_mix = np.unique(sam_pmask_2d_mix, return_inverse=True) # re-index
            sam_pmask_2d_mix[sam_pmask_2d_mix>0] += N_sam # not overlap id
            mix_sam_pred = True
            

            
        binary_mask = None
        binary_mask_orig = torch.zeros(len(points))
        out_points = []
        out_pts_semantic_mask, out_pts_instance_mask = [], []
        block_orig, block_mix = [], []
            
        if self.mode == 'V': # vertical cut and mix frustums
            num_areas = np.random.choice(self.v_num_areas, size=1)[0]
            area_splits = divide_point_cloud_yaw(num_areas)
            
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:, 0])
        elif self.mode == 'H': # horizontal cut and mix frustums
            num_areas = np.random.choice(self.h_num_areas, size=1)[0]
            area_splits = divide_point_cloud_pitch(num_areas, self.pitch_angles)
            
            rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
            pitch = torch.atan2(points.coord[:, 2], rho)
            pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5, self.pitch_angles[1] - 1e-5)

            mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 + mix_points.coord[:, 1]**2)
            mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
            mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5, self.pitch_angles[1] - 1e-5)
        else:
            raise ValueError(f'Invalid mode for mixing: {self.mode}')   
        
        for i, area_split in enumerate(area_splits):
            start_angle = area_split[0]
            end_angle = area_split[1]
            if i % 2 == 0:  # pick from original point cloud
                idx = split_yaw(yaw, start_angle, end_angle) if self.mode == 'V' else (pitch > start_angle) & (pitch <= end_angle)
                selected_points = points[idx]
                
                out_points.append(points[idx])
                block_orig.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask_2d.append(sam_pmask_2d[idx.numpy()])
                    out_sam_pscore_2d.append(sam_pscore_2d[idx.numpy()])
                if binary_mask == None:
                    binary_mask = torch.ones(selected_points.shape[0], dtype=torch.uint8)
                else:
                    binary_mask = torch.cat([binary_mask, torch.ones(selected_points.shape[0], dtype=torch.uint8)])  # Mark as original
                binary_mask_orig[idx] = 1  
                
            else:  # pickle from mixed point cloud
                idx = split_yaw(mix_yaw, start_angle, end_angle) if self.mode == 'V' else (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                selected_points = mix_points[idx]
                
                out_points.append(mix_points[idx])
                block_mix.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])

                if mix_panoptic:
                    out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask_2d.append(sam_pmask_2d_mix[idx.numpy()])
                    out_sam_pscore_2d.append(sam_pscore_2d_mix[idx.numpy()])
                binary_mask = torch.cat([binary_mask, torch.zeros(selected_points.shape[0], dtype=torch.uint8)])  # Mark as modified     
        
        points = points.cat(out_points)
        pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        if mix_panoptic:
            pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
        if mix_sam_pred:
            sam_pmask_2d = np.concatenate(out_sam_pmask_2d, axis=0)
            sam_pscore_2d = np.concatenate(out_sam_pscore_2d, axis=0)
        
        ########## image augmentation ##########
        if self.img_aug:
            out_im_list = []
            image_list_orig = input_dict['img']
            image_list_mix = mix_results['img']
            image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
            image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
            
            lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
            
            img_size = input_dict['img_shape'][:2] # (360, 630)
            ori_size = input_dict['ori_shape'] # (900, 1600)
            img_scale = input_dict['scale_factor'][0] # 0.4
            assert int(img_size[0]) == int(ori_size[0] * img_scale)
            
            for v in range(len(image_list_orig)):
                im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                lidar2img_mix_v = lidar2img_mix[v] # @ torch.inverse(aug_mat)
                for i in range(len(block_mix)):
                    pblock = block_mix[i].coord
                    points_img_b, mask_b = proj_lidar2img(pblock, lidar2img_mix_v, 
                                                        img_size=(ori_size[1], ori_size[0]),  # w, h
                                                        min_dist=1.0)
                    points_img_b = points_img_b*img_scale
                    if self.mode == 'V':
                        mixed_img, block_mask = merge_images_yaw_torch(im_orig, im_mix, points_img_b) 
                    elif self.mode == 'H':
                        mixed_img, block_mask = merge_images_pitch_torch(im_orig, im_mix, points_img_b)
                    else:
                        raise ValueError(f'Invalid mode for mixing: {self.mode}')  
                    im_orig = mixed_img
                out_im_list.append(im_orig)
        ########## image augmentation ##########
        
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            assert sam_pmask_2d.shape[0] == pts_semantic_mask.shape[0]
            assert sam_pscore_2d.shape[0] == pts_semantic_mask.shape[0]
            input_dict['sam_pmask_2d'] = sam_pmask_2d
            input_dict['sam_pscore_2d'] = sam_pscore_2d
        
        assert binary_mask.shape[0] == points.coord.shape[0]
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig  
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.pie_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'yaw_angles={self.yaw_angles}, '
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

# V2: for decoupled instance copy and rotate-paste.
@TRANSFORMS.register_module(force=True)
class _PieMix_Inst_V2(BaseTransform):
    """PierMix data augmentation. 
    """
    def __init__(self,
                 img_aug: bool,
                 instance_classes: List[int],
                 rotate: bool,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.rotate = rotate
        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
            
    def pie_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """
        copy instance points from mixed point cloud and rotate-paste them to the original point cloud.
        """
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            pts_instance_mask = input_dict['pts_instance_mask']
            N = pts_instance_mask.max()
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += N # not overlap id
            # mix_instance_mask += (1000<<16) # not overlap id
            mix_panoptic = True
            
        binary_mask = torch.ones(len(points))
        binary_mask_orig = torch.ones(len(points))
        out_points = []
        out_pts_semantic_mask, out_pts_instance_mask = [], []
        block_orig, block_mix = [], []
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            assert sam_pmask_2d_mix.dtype == np.int64
            sam_pmask_2d_mix[sam_pmask_2d_mix>0] += (1000<<16) # not overlap id
            sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            out_sam_pmask_2d, out_sam_pscore_2d = [], []
            mix_sam_pred = True
            
        ## 1. copy instance points from new scan
        instance_points, instance_pts_semantic_mask = [], []
        if mix_panoptic:
            instance_pts_instance_mask = []
        if mix_sam_pred:
            instance_pts_sam_pmask_2d, instance_pts_sam_pscore_2d = [], []
        for instance_class in self.instance_classes:
            mix_idx = mix_pts_semantic_mask == instance_class
            instance_points.append(mix_points[mix_idx])
            instance_pts_semantic_mask.append(mix_pts_semantic_mask[mix_idx])
            if mix_panoptic:
                instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
            if mix_sam_pred:
                instance_pts_sam_pmask_2d.append(sam_pmask_2d_mix[mix_idx])
                instance_pts_sam_pscore_2d.append(sam_pscore_2d_mix[mix_idx])
        # print(f'len of instance_points: {len(instance_points)}')
        instance_points = mix_points.cat(instance_points)
        # print(f'instance_points shape: {instance_points.coord.shape}')
        instance_pts_semantic_mask = np.concatenate(instance_pts_semantic_mask, axis=0)
        if mix_panoptic:
            instance_pts_instance_mask = np.concatenate(instance_pts_instance_mask, axis=0) 
        if mix_sam_pred:
            instance_pts_sam_pmask_2d = np.concatenate(instance_pts_sam_pmask_2d, axis=0)
            instance_pts_sam_pscore_2d = np.concatenate(instance_pts_sam_pscore_2d, axis=0)
            
        # update points to the original scan
        points = points.cat([points, instance_points])
        pts_semantic_mask = np.concatenate((pts_semantic_mask, instance_pts_semantic_mask), axis=0)
        if mix_panoptic:
            pts_instance_mask = np.concatenate((pts_instance_mask, instance_pts_instance_mask), axis=0)
        if mix_sam_pred:
            sam_pmask_2d = np.concatenate((sam_pmask_2d, instance_pts_sam_pmask_2d), axis=0)
            sam_pscore_2d = np.concatenate((sam_pscore_2d, instance_pts_sam_pscore_2d), axis=0)
        binary_mask = torch.cat([binary_mask, torch.zeros(instance_points.coord.shape[0])])  # Mark rotated-pasted points as modified
        binary_mask_orig = binary_mask_orig # pasting does not change the original points
        # print(f' instance shape: {instance_pts_instance_mask.shape}')
        if instance_pts_instance_mask.shape[0] < 1:
            # print(f'$$$$$$$$$$$$$$$$$$$$ no inst found: point shape {points.coord.shape}, instance shape {instance_pts_instance_mask.shape}, instance shape {instance_points.coord.shape}')
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
            assert points.coord.shape[0] == pts_semantic_mask.shape[0]
            return input_dict
        # N = instance_pts_instance_mask.max()
        # paste instances' images to the original image
        out_im_list, id2img = update_inst_img(input_dict, mix_results,  instance_points.coord, 
                                              instance_pts_instance_mask, instance_pts_semantic_mask, 
                                               source='newscan')
        input_dict['img'] = out_im_list
        
        ## 2. rotate-copy 
        if self.rotate:
            angle_list = [
                np.random.random() * np.pi * 2 / 3,
                (np.random.random() + 1) * np.pi * 2 / 3
            ]
            ro_coords, ro_instlabels, ro_semlabels = [], [], []
            
            n = np.unique(instance_pts_instance_mask).shape[0]
            # _, re_inst = np.unique(block_ro_instlabel, return_inverse=True)
            for k, angle in enumerate(angle_list):
                block_ro = instance_points.clone()
                block_ro.rotate(angle, axis=2) # rotate around z-axis
                # block_ro_instlabel = instance_pts_instance_mask.copy() #NOTE: should be new instance ID
                re_inst = instance_pts_instance_mask.copy() #  + n * (k+1)
                block_ro_semlabel = instance_pts_semantic_mask.copy()
                

                #NOTE: instance selection
                # collison check 
                # class-wise sampling
                # ....
            
                # paste the rotated-copied instances' images to the original image
                out_im_list, ro_imgs = update_inst_img(input_dict, mix_results, block_ro.coord, re_inst, block_ro_semlabel, 
                                                    source='presaved', id2img=id2img)
                input_dict['img'] = out_im_list
                # update points to the original scan  
                ro_coords.append(block_ro)
                ro_semlabels.append(block_ro_semlabel)
                if mix_panoptic:
                    ro_instlabels.append(re_inst)
                print(f'###### remember to update the sam_pred mask')
                
            ro_coords = points.cat(ro_coords)
            points = points.cat([points, ro_coords])
            ro_semlabels = np.concatenate(ro_semlabels, axis=0)
            pts_semantic_mask = np.concatenate((pts_semantic_mask, ro_semlabels), axis=0)
            if mix_panoptic:
                ro_instlabels = np.concatenate(ro_instlabels, axis=0)
                pts_instance_mask = np.concatenate((pts_instance_mask, ro_instlabels), axis=0)
            
            binary_mask = torch.cat([binary_mask, torch.zeros(ro_coords.shape[0])])  # Mark rotated-pasted points as modified
        #TODO: modify the sam_pred mask
        
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            assert sam_pmask_2d.shape[0] == points.coord.shape[0]
            assert sam_pmask_2d.shape[0] == pts_semantic_mask.shape[0]
            assert sam_pscore_2d.shape[0] == pts_semantic_mask.shape[0]
            input_dict['sam_pmask_2d'] = sam_pmask_2d
            input_dict['sam_pscore_2d'] = sam_pscore_2d
        
        assert binary_mask.shape[0] == points.coord.shape[0]
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        # if 'augment_mask' in input_dict:
        #     input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
        #     input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig  
        # else:
        input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
        input_dict['augment_mask_orig'] = binary_mask_orig
        
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.pie_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'yaw_angles={self.yaw_angles}, '
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

# V3: search empty area to paste the rotated instances.
@TRANSFORMS.register_module(force=True)
class _PieMix_Inst_V3(BaseTransform):
    """PierMix data augmentation. 
    """
    def __init__(self,
                 img_aug: bool,
                 instance_classes: List[int],
                 instance_weights: List[float],
                 num_add_instances: int = 10,
                 point_attr: str='LIDAR',
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.instance_weights = instance_weights
        if isinstance(self.instance_weights, list):
            self.instance_weights = np.array(self.instance_weights)
        self.num_add_instances = num_add_instances
        self.pack_attr = get_points_type(point_attr) # pack tensor to LIDARPoints
        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)
            
    def pie_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """
        copy instance points from mixed point cloud and rotate-paste them to the original point cloud.
        """
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        binary_mask = torch.ones(len(points))
        binary_mask_orig = torch.ones(len(points))
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            pts_instance_mask = input_dict['pts_instance_mask']
            _, pts_instance_mask = np.unique(pts_instance_mask, return_inverse=True) # re-index
            N = pts_instance_mask.max()
            mix_instance_mask = mix_results['pts_instance_mask']
            _, mix_instance_mask = np.unique(mix_instance_mask, return_inverse=True) # re-index
            # print(f'orig instance num: {N}, new instance num: {mix_instance_mask.max()}')
            mix_panoptic = True
            
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            # sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            # assert sam_pmask_2d_mix.dtype == np.int64
            # sam_pmask_2d_mix[sam_pmask_2d_mix>0] += (1000<<16) # not overlap id
            # sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            assert sam_pmask_2d.dtype == np.int64
            N_sam = sam_pmask_2d.max()
            mix_sam_pred = True
            
        ## 1. copy instance points from new scan
        instance_points, instance_pts_semantic_mask = [], []
        if mix_panoptic:
            instance_pts_instance_mask = []
        for instance_class in self.instance_classes:
            mix_idx = mix_pts_semantic_mask == instance_class
            instance_points.append(mix_points[mix_idx])
            instance_pts_semantic_mask.append(mix_pts_semantic_mask[mix_idx])
            if mix_panoptic:
                instance_pts_instance_mask.append(mix_instance_mask[mix_idx])
        
        instance_points = mix_points.cat(instance_points)
        instance_pts_semantic_mask = np.concatenate(instance_pts_semantic_mask, axis=0)
        if mix_panoptic:
            instance_pts_instance_mask = np.concatenate(instance_pts_instance_mask, axis=0) 
        
        if instance_pts_instance_mask.shape[0] < 1:
            return input_dict
        ## 2. register the instance images
        if self.img_aug:
            _, id2img = update_inst_img(input_dict, mix_results,  instance_points.coord, 
                                              instance_pts_instance_mask, instance_pts_semantic_mask, 
                                               source='newscan')
        
        ## 3. rotate-copy 
        ## v1: set all class weights to 1.
        # tar_cls_lst = np.unique(instance_pts_semantic_mask)
        # instance_weights = np.ones(len(tar_cls_lst))/len(tar_cls_lst) #NOTE: V1: set all class weights to 1.
        # print('v1: set all class weights to 1.')
        ## v2: set class weights according to the instance number
        tar_cls_lst = np.unique(instance_pts_semantic_mask)
        nusc_cls_prob = self.instance_weights
        nusc_cls_prob= nusc_cls_prob[tar_cls_lst-1]
        nusc_cls_prob = nusc_cls_prob/np.sum(nusc_cls_prob)
        assert len(tar_cls_lst) == len(nusc_cls_prob), 'class and weight mismatch' 
        
        sem2inst_map = sem2inst(instance_pts_semantic_mask, instance_pts_instance_mask, tar_cls_lst)

        inst2scene_augmentor = InstanceInsertor(sem2inst_map, tar_cls_lst, 
                                                     nusc_cls_prob, add_num=self.num_add_instances,
                                                    inst_noise=True, inst_part_remv=True)
        aug_mask, add_xyz, add_sem, add_inst, add_raw_ids = inst2scene_augmentor.instance_aug(
                                                    points.tensor, pts_semantic_mask, pts_instance_mask, 
                                                    instance_points.tensor, instance_pts_instance_mask)
        if not isinstance(aug_mask, torch.Tensor):
            aug_mask = torch.tensor(aug_mask)
            add_xyz = [torch.tensor(xyz) for xyz in add_xyz]
            # add_sem = [torch.tensor(sem) for sem in add_sem]
            # add_inst = [torch.tensor(inst) for inst in add_inst]
        add_raw_ids = [torch.tensor(ids) for ids in add_raw_ids]
        
        # update to the original scan   
        ## filter out the points that are behind the added instances
        assert points.shape[0] == pts_semantic_mask.shape[0]
        points = points[torch.where(aug_mask>0)[0]]
        N_raw_points = points.shape[0]
        pts_semantic_mask = pts_semantic_mask[torch.where(aug_mask>0)[0].numpy()]
        if mix_panoptic:
            pts_instance_mask = pts_instance_mask[torch.where(aug_mask>0)[0].numpy()]
        if mix_sam_pred:
            sam_pmask_2d = sam_pmask_2d[torch.where(aug_mask>0)[0].numpy()]
            sam_pscore_2d = sam_pscore_2d[torch.where(aug_mask>0)[0].numpy()]
            
        assert points.shape[0] == pts_semantic_mask.shape[0]
        ## add the new instances
        if len(add_xyz) < 1:
            return input_dict
        add_xyz = torch.cat(add_xyz, dim=0)
        add_xyz = self.pack_attr(add_xyz, points_dim=add_xyz.shape[-1], attribute_dims=None)
        points = points.cat([points, add_xyz])
        add_sem = np.concatenate(add_sem, axis=0)
        # print(f'instance sematic copied {np.unique(add_sem)}')
        pts_semantic_mask = np.concatenate((pts_semantic_mask, add_sem), axis=0)
        assert add_xyz.shape[0] == add_sem.shape[0]
        assert points.shape[0] == pts_semantic_mask.shape[0]
        if mix_panoptic:
            add_inst = np.concatenate(add_inst, axis=0)
            pts_instance_mask = np.concatenate((pts_instance_mask, add_inst), axis=0)
        if mix_sam_pred:
            _, add_sam = np.unique(add_inst, return_inverse=True)
            # print(f'BEFORE: sam id copied {np.unique(add_sam)}, source id {np.unique(sam_pmask_2d)}')
            add_sam[add_sam>0] += N_sam
            # print(f'AFTER: sam id copied {np.unique(add_sam)}')
            sam_pmask_2d = np.concatenate((sam_pmask_2d, add_sam), axis=0)
            sam_pscore_2d = np.concatenate((sam_pscore_2d, np.ones(add_sam.shape[0])), axis=0)
        add_raw_ids = torch.cat(add_raw_ids, dim=0)
        binary_mask = aug_mask.clone()
        binary_mask = torch.cat([torch.ones(N_raw_points), torch.zeros(add_xyz.shape[0])])
        binary_mask_orig = aug_mask.clone()
        
        ## 4. paste images
        assert add_raw_ids.shape[0] == add_sem.shape[0], f'{add_raw_ids.shape[0]} != {add_sem.shape[0]}'
        if self.img_aug:
            out_im_list, _ = update_inst_img(input_dict, mix_results, add_xyz.coord, add_inst, add_sem, 
                                            source='presaved', id2img=id2img, draw_instlabel=add_raw_ids)
            input_dict['images'] = out_im_list
        
        ## 5. pack the results
        assert binary_mask.shape[0] == points.shape[0]
        assert points.shape[0] == pts_semantic_mask.shape[0]
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        if mix_panoptic:
            input_dict['pts_instance_mask'] = pts_instance_mask
        if mix_sam_pred:
            assert sam_pmask_2d.shape[0] == pts_semantic_mask.shape[0]
            assert sam_pscore_2d.shape[0] == pts_semantic_mask.shape[0]
            input_dict['sam_pmask_2d'] = sam_pmask_2d
            input_dict['sam_pscore_2d'] = sam_pscore_2d
        
        input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
        input_dict['augment_mask_orig'] = binary_mask_orig
        
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.pie_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'instance_weights={self.instance_weights}, '
        repr_str += f'num_add_instances={self.num_add_instances}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str



@TRANSFORMS.register_module(force=True)
class _LaserMix_MM(BaseTransform):
    """LaserMix data augmentation.

    The lasermix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Divide the point cloud into several regions according to pitch
           angles and combine the areas crossly.

    Required Keys: 

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        num_areas (List[int]): A list of area numbers will be divided into.
        pitch_angles (Sequence[float]): Pitch angles used to divide areas.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def laser_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        binary_mask = None
        binary_mask_orig = torch.zeros(len(points), dtype=torch.uint8)

        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5,
                            self.pitch_angles[1] - 1e-5)

        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 +
                             mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0],
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []

        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            out_pts_instance_mask = []
            mix_panoptic = True

        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                selected_points = points[idx]
                
                out_points.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
                if binary_mask == None:
                    binary_mask = torch.ones(selected_points.shape[0], dtype=torch.uint8)
                else:
                    binary_mask = torch.cat([binary_mask, torch.ones(selected_points.shape[0], dtype=torch.uint8)])  # Mark as original
                binary_mask_orig[idx] = 1  
                
            else:  # pickle from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                selected_points = mix_points[idx]
                
                out_points.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
                
                binary_mask = torch.cat([binary_mask, torch.zeros(selected_points.shape[0], dtype=torch.uint8)])  # Mark as modified

        out_points = points.cat(out_points)
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask

        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        
        if mix_panoptic:
            out_pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
            input_dict['pts_instance_mask'] = out_pts_instance_mask
        
        #############################
        # check whether use LaserMix
        # print('********************* USING LASERMIX NOW *********************')
        #############################
        
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.laser_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module(force=True)
class _LaserMix_MM_IMG(BaseTransform):
    """LaserMix data augmentation.

    The lasermix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Divide the point cloud into several regions according to pitch
           angles and combine the areas crossly.

    Required Keys: 

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)
    - images (dict): meta data of images: {'VIEW_NAME': {'img_path': str, 'lidar2img': np.ndarray}}
    - img (list): list of view images in np.ndarray
    
    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - img (list): list of view images in np.ndarray

    Args:
        num_areas (List[int]): A list of area numbers will be divided into.
        pitch_angles (Sequence[float]): Pitch angles used to divide areas.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 img_aug: bool,
                 num_areas: List[int],
                 pitch_angles: Sequence[float],
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        self.img_aug = img_aug
        assert is_list_of(num_areas, int), \
            'num_areas should be a list of int.'
        self.num_areas = num_areas

        assert len(pitch_angles) == 2, \
            'The length of pitch_angles should be 2, ' \
            f'but got {len(pitch_angles)}.'
        assert pitch_angles[1] > pitch_angles[0], \
            'pitch_angles[1] should be larger than pitch_angles[0].'
        self.pitch_angles = pitch_angles

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def laser_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']
        
        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']
        binary_mask = None
        binary_mask_orig = torch.zeros(len(points), dtype=torch.uint8)


        rho = torch.sqrt(points.coord[:, 0]**2 + points.coord[:, 1]**2)
        pitch = torch.atan2(points.coord[:, 2], rho)
        pitch = torch.clamp(pitch, self.pitch_angles[0] + 1e-5,
                            self.pitch_angles[1] - 1e-5)

        mix_rho = torch.sqrt(mix_points.coord[:, 0]**2 +
                             mix_points.coord[:, 1]**2)
        mix_pitch = torch.atan2(mix_points.coord[:, 2], mix_rho)
        mix_pitch = torch.clamp(mix_pitch, self.pitch_angles[0] + 1e-5,
                                self.pitch_angles[1] - 1e-5)

        num_areas = np.random.choice(self.num_areas, size=1)[0]
        angle_list = np.linspace(self.pitch_angles[1], self.pitch_angles[0],
                                 num_areas + 1)
        out_points = []
        out_pts_semantic_mask = []
        block_orig, block_mix = [], []
        
        mix_panoptic = False
        if 'pts_instance_mask' in mix_results:
            mix_instance_mask = mix_results['pts_instance_mask']
            mix_instance_mask += (1000<<16) # not overlap id
            pts_instance_mask = input_dict['pts_instance_mask']
            out_pts_instance_mask = []
            mix_panoptic = True
        
        mix_sam_pred = False
        if 'sam_pmask_2d' in mix_results:
            sam_pmask_2d_mix = mix_results['sam_pmask_2d']
            assert sam_pmask_2d_mix.dtype == np.int64
            sam_pmask_2d_mix[sam_pmask_2d_mix>0] += (1000<<16) # not overlap id
            sam_pscore_2d_mix = mix_results['sam_pscore_2d']
            sam_pmask_2d = input_dict['sam_pmask_2d']
            sam_pscore_2d = input_dict['sam_pscore_2d']
            out_sam_pmask_2d, out_sam_pscore_2d = [], []
            mix_sam_pred = True
            
        mix_tgt_pl = False
        if 'UDA' in mix_results:
            pl_psem = input_dict['UDA']['tgt_pl_psem']
            pl_pins = input_dict['UDA']['tgt_pl_pins']
            pl_pscore = input_dict['UDA']['tgt_pl_pscore']
            pl_psem_mix = mix_results['UDA']['tgt_pl_psem']
            pl_pins_mix = mix_results['UDA']['tgt_pl_pins']
            pl_pscore_mix = mix_results['UDA']['tgt_pl_pscore']
            out_pl_psem, out_pl_pins, out_pl_pscore = [], [], []
            mix_tgt_pl = True
            
        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1] / 180 * np.pi
            end_angle = angle_list[i] / 180 * np.pi
            if i % 2 == 0:  # pick from original point cloud
                idx = (pitch > start_angle) & (pitch <= end_angle)
                selected_points = points[idx]
                
                out_points.append(points[idx])
                block_orig.append(points[idx])
                out_pts_semantic_mask.append(pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(pts_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask_2d.append(sam_pmask_2d[idx.numpy()])
                    out_sam_pscore_2d.append(sam_pscore_2d[idx.numpy()])
                if mix_tgt_pl:
                    out_pl_psem.append(pl_psem[idx.numpy()])
                    out_pl_pins.append(pl_pins[idx.numpy()])
                    out_pl_pscore.append(pl_pscore[idx.numpy()])
                if binary_mask == None:
                    binary_mask = torch.ones(selected_points.shape[0], dtype=torch.uint8)
                else:
                    binary_mask = torch.cat([binary_mask, torch.ones(selected_points.shape[0], dtype=torch.uint8)])  # Mark as original
                binary_mask_orig[idx] = 1  
                
            else:  # pickle from mixed point cloud
                idx = (mix_pitch > start_angle) & (mix_pitch <= end_angle)
                selected_points = mix_points[idx]
                
                out_points.append(mix_points[idx])
                block_mix.append(mix_points[idx])
                out_pts_semantic_mask.append(
                    mix_pts_semantic_mask[idx.numpy()])
                if mix_panoptic:
                    out_pts_instance_mask.append(mix_instance_mask[idx.numpy()])
                if mix_sam_pred:
                    out_sam_pmask_2d.append(sam_pmask_2d_mix[idx.numpy()])
                    out_sam_pscore_2d.append(sam_pscore_2d_mix[idx.numpy()])
                if mix_tgt_pl:
                    out_pl_psem.append(pl_psem_mix[idx.numpy()])
                    out_pl_pins.append(pl_pins_mix[idx.numpy()])
                    out_pl_pscore.append(pl_pscore_mix[idx.numpy()])
                binary_mask = torch.cat([binary_mask, torch.zeros(selected_points.shape[0], dtype=torch.uint8)])  # Mark as modified

        # save pts blocks
        # input_dict['pts_blocks_orig'] = block_orig
        # input_dict['pts_blocks_mix'] = block_mix
        # input_dict['pts_semantic_mask_blocks'] = out_pts_semantic_mask
        
        ########## image augmentation ##########
        if self.img_aug:
            out_im_list = []
            image_list_orig = input_dict['img']
            image_list_mix = mix_results['img']
            image_list_orig = [torch.tensor(img) for img in image_list_orig] if isinstance(image_list_orig[0], np.ndarray) else image_list_orig
            image_list_mix  = [torch.tensor(img) for img in image_list_mix] if isinstance(image_list_mix[0], np.ndarray) else image_list_mix
            
            img_size = input_dict['img_shape'][:2] # (360, 630)
            ori_size = input_dict['ori_shape'] # (900, 1600)
            img_scale = input_dict['scale_factor'][0] # 0.4
            assert int(img_size[0]) == int(ori_size[0] * img_scale)
            
            lidar2img_orig, lidar2img_mix = input_dict['lidar2img'], mix_results['lidar2img']
            # if isinstance(lidar2img_orig, dict):
            #     lidar2img_orig = list(lidar2img_orig.values())
            #     lidar2img_mix  = list(lidar2img_mix.values())
            
            for v in range(len(image_list_orig)):
                im_orig, im_mix = image_list_orig[v].clone(), image_list_mix[v].clone()
                for i in range(len(block_mix)):
                    pblock = block_mix[i].coord
                    points_img_b, mask_b = proj_lidar2img(pblock, lidar2img_mix[v], 
                                                        img_size=(ori_size[1], ori_size[0]), 
                                                        min_dist=1.0)
                    points_img_b = points_img_b * img_scale
                    mixed_img, block = merge_images_pitch_torch(im_orig, im_mix, points_img_b)   
                    im_orig = mixed_img
                ########## save image for debug ##########
                # sample_token = input_dict['token']
                # view = list(input_dict['images'].keys())[v]
                # denorm_img = (im_orig - im_orig.min()) / (im_orig.max() - im_orig.min()) * 255
                # denorm_img = denorm_img.astype(np.uint8)
                # cv2.imwrite(f'misc/rendered_point_mmaug/{sample_token}_{view}_laser.png', denorm_img)
                ########## save image for debug ##########
                out_im_list.append(im_orig)
        ########## image augmentation ##########
            
                
            
        # print(f'1********point shape {points.shape}, binary mask shape {binary_mask.shape}')
        out_points = points.cat(out_points)
        # print(f'2********point shape {out_points.shape}')
    
        out_pts_semantic_mask = np.concatenate(out_pts_semantic_mask, axis=0)
        input_dict['points'] = out_points
        input_dict['pts_semantic_mask'] = out_pts_semantic_mask
        # input_dict['binary_mask'] = binary_mask
        input_dict['lidars2imgs_mix_torch'] = mix_results['lidar2img']
        
        assert binary_mask.shape[0] == out_points.coord.shape[0]
        # print(torch.count_nonzero(binary_mask), torch.count_nonzero(binary_mask_orig))
        assert torch.count_nonzero(binary_mask) == torch.count_nonzero(binary_mask_orig)
        if 'augment_mask' in input_dict:
            input_dict['augment_mask'] = input_dict['augment_mask'] & binary_mask  # intersaction the binary mask
            input_dict['augment_mask_orig'] = input_dict['augment_mask_orig'] & binary_mask_orig
        else:
            input_dict['augment_mask'] = binary_mask # 1 for original, 0 for modified
            input_dict['augment_mask_orig'] = binary_mask_orig
        if mix_panoptic:
            out_pts_instance_mask = np.concatenate(out_pts_instance_mask, axis=0)
            input_dict['pts_instance_mask'] = out_pts_instance_mask
        if mix_sam_pred:
            out_sam_pmask_2d = np.concatenate(out_sam_pmask_2d, axis=0)
            out_sam_pscore_2d = np.concatenate(out_sam_pscore_2d, axis=0)
            assert out_sam_pmask_2d.shape[0] == out_points.shape[0]
            assert out_sam_pscore_2d.shape[0] == out_points.shape[0]
            input_dict['sam_pmask_2d'] = out_sam_pmask_2d
            input_dict['sam_pscore_2d'] = out_sam_pscore_2d
        if mix_tgt_pl:
            out_pl_psem = np.concatenate(out_pl_psem, axis=0)
            out_pl_pins = np.concatenate(out_pl_pins, axis=0)
            out_pl_pscore = np.concatenate(out_pl_pscore, axis=0)
            assert out_pl_psem.shape[0] == out_points.shape[0]
            assert out_pl_pins.shape[0] == out_points.shape[0]
            assert out_pl_pscore.shape[0] == out_points.shape[0]
            input_dict['UDA']['tgt_pl_psem'] = out_pl_psem
            input_dict['UDA']['tgt_pl_pins'] = out_pl_pins
            input_dict['UDA']['tgt_pl_pscore'] = out_pl_pscore
        if self.img_aug:
            input_dict['img'] = out_im_list
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """LaserMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through LaserMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset)) #TODO: not used for debug
        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})  #TODO: not used for debug
            # before lasermix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')
        
        input_dict = self.laser_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_areas={self.num_areas}, '
        repr_str += f'pitch_angles={self.pitch_angles}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str



# #TODO-YINING: not finished yet
# @TRANSFORMS.register_module(force=True)
# class SuperPointPartitionTransform(BaseTransform):
#     def __init__(self, spg_file_path, pre_transform=None, prob=1.0):
#         self.spg_file_path = spg_file_path
#         self.pre_transform = pre_transform
#         self.prob = prob

#     def read_incomponent(self, file_name):
#         """read the mapping relation(components and in_component) only."""
#         data_file = h5py.File(file_name, 'r')
#         in_component = np.array(data_file["in_component"], dtype='uint32')
#         return in_component

#     def get_spg_file_path(self, lidar_path, spg_prefix='/mnt/workspace/superpoint_graph/dataset/skitti/superpoint_graphs/'):
#         '''get pre-processed spg file path from corresponding lidar_path and prefix.'''
#         sequence = os.path.dirname(lidar_path).split('/velodyne')[0][-2:]
#         file_name = os.path.splitext(os.path.basename(lidar_path))[0]
#         spg_file = os.path.join(spg_prefix, sequence, file_name + '.h5')
#         return spg_file

#     def sp_partition(self, xyz, k_nn_geof=45, k_nn_adj=10, lambda_edge_weight=1., reg_strength=0.03, d_se_max=0):
#         #---compute 10 nn graph-------
#         graph_nn, target_fea = compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof)

#         #---compute geometric features-------
#         geof = libply_c.compute_geof(xyz, target_fea, k_nn_geof).astype('float32')
#         # del target_fea

#         # features = np.hstack((geof, rgb/255.)).astype('float32')
#         features = geof
#         features[:,3] = 2. * features[:,3] 
#         graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')

#         components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"], graph_nn["edge_weight"], reg_strength)
#         components = np.array(components, dtype = 'object')
#         # graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, labels, n_labels)
        
#         return components, in_component


#     def transform(self, results):
#         if np.random.rand() > self.prob:
#             return results

#         assert 'points' in results, 'Points are needed for SuperPoint partition'
#         # assert 'pts_semantic_mask' in results, 'pts_semantic_mask is needed for SuperPoint partition'

#         if self.spg_file_path and os.path.exists(self.spg_file_path):
#             # Read SPG data from file
#             spg_file = xxx
#             spg_data = self.read_incomponent(spg_file)
#         else:
#             # Perform SP Partition
#             spg_data = self.sp_partition()

#         # Update results with SPG data
#         results['spg_data'] = DC(spg_data, stack=False)

#         if self.pre_transform is not None:
#             results = self.pre_transform(results)

#         return results

#     def __repr__(self) -> str:
#         """str: Return a string that describes the module."""
#         repr_str = self.__class__.__name__
#         repr_str += f'(num_areas={self.num_areas}, '
#         repr_str += f'pitch_angles={self.pitch_angles}, '
#         repr_str += f'pre_transform={self.pre_transform}, '
#         repr_str += f'prob={self.prob})'
#         return repr_str
