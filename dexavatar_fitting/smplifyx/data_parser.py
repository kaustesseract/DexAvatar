# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import pickle

import json

from collections import namedtuple

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


from utils import smpl_to_openpose

Keypoints = namedtuple('Keypoints',
                       ['keypoints', 'gender_gt', 'gender_pd'])

Keypoints.__new__.__defaults__ = (None,) * len(Keypoints._fields)


def create_dataset(indp_sign_segment=None, dataset='openpose', data_folder='data', indp_sign_class=None, **kwargs):
    if dataset.lower() == 'openpose':
        return OpenPose(data_folder, indp_sign_segment, indp_sign_class, **kwargs)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=False):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    keypoints = []

    gender_pd = []
    gender_gt = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                person_data['face_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    person_data['face_keypoints_2d'],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        if 'gender_pd' in person_data:
            gender_pd.append(person_data['gender_pd'])
        if 'gender_gt' in person_data:
            gender_gt.append(person_data['gender_gt'])

        keypoints.append(body_keypoints)

    return Keypoints(keypoints=keypoints, gender_pd=gender_pd,
                     gender_gt=gender_gt)


class OpenPose(Dataset):

    NUM_BODY_JOINTS = 25
    NUM_HAND_JOINTS = 20

    def __init__(self, data_folder, indp_sign_segment=None, indp_sign_class=None, img_path='images', 
                 keyp_folder='keypoints',
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 use_face_contour=False,
                 openpose_format='coco25',
                 **kwargs):
        super(OpenPose, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.use_face_contour = use_face_contour
        self.indp_sign_class = indp_sign_class
        self.prev_2D_keypoints = None
        self.prev_3D_keypoints = None
        self.prev_hand_rotations = None

        self.openpose_format = openpose_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.img_folder = img_path
        self.keyp_folder = osp.join(data_folder, keyp_folder)

        img_fns = [fn for fn in os.listdir(self.img_folder)
           if (fn.endswith('.png') or fn.endswith('.jpg')) and not fn.startswith('.')]

        img_fns.sort(key=lambda fn: int(osp.splitext(fn)[0].split('_')[-1]))

        self.img_paths = [osp.join(self.img_folder, fn) for fn in img_fns]

        start_id, end_id = indp_sign_segment[0], indp_sign_segment[1]

        def filename_to_int(fn):
            # fn might be "low_164.png" → we want 164 as an integer
            base = os.path.basename(fn)            # "low_164.png"
            number_str = os.path.splitext(base)[0]  # "low_164"
            return int(number_str.split("_")[-1])   # 164

        # Now build a new, filtered list
        selected = []

        for full_path in self.img_paths:
            n = filename_to_int(full_path)
            if start_id <= n <= end_id:
                selected.append(full_path)

        # If you need the index‐range in the original list:
        all_ids = [filename_to_int(p) for p in self.img_paths]
        start_idx = next(i for i,v in enumerate(all_ids) if v == start_id)
        end_idx   = next(i for i,v in enumerate(all_ids) if v == end_id)

        self.img_paths = self.img_paths[start_idx : end_idx+1]

        #print("Segmented image paths: ", self.img_paths)

        self.cnt = 0
        self.sapiens_path = os.path.join(data_folder, 'sapiens.pkl')
        with open(self.sapiens_path, 'rb') as f:
            self.sapiens = pickle.load(f)
        hamer_path = os.path.join(data_folder, 'hamer', 'hamer.pkl')
        with open(hamer_path, 'rb') as f:
            self.hamer = pickle.load(f)

        temp = []

        for x in self.img_paths:

            base = x.split('/')[-1][:-4]   
        
            if x.split('/')[-1] not in self.hamer:
                continue
            
            if self.hamer[x.split('/')[-1]][0]['pred_keypoints_2d'].shape[0] < 1:
                continue
            
            smplx_param_path = os.path.join(data_folder, 'smplerx/smplx', f'{base}.pkl')
            
            if not os.path.exists(smplx_param_path):
                continue

            temp.append(x)
        
        self.img_paths = sorted(temp)

        if self.indp_sign_class == "0":
            self.active_side, self.w_left, self.w_right = self._compute_motion_scores_for_clip()
            self.one_hand_is_right = (self.active_side == 'right')

        self.avg_shape = np.load(os.path.join(data_folder, 'mean_shape_smplx.npy'))
        self.data_folder = data_folder


    def _to_numpy(self, x):
        import numpy as np, torch
        if isinstance(x, np.ndarray):
            return x
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _ensure_float(x):
        """Safely extract a Python float from torch/np/scalar."""
        try:
            return float(x)
        except Exception:
            import numpy as np, torch
            if torch.is_tensor(x):
                return float(x.detach().cpu().item())
            if isinstance(x, np.ndarray):
                return float(x.reshape(-1)[0])
            return float(x)

    def _unnorm_hand_kp2d(self, kp2d, cx, cy, box_size, is_right):
        """
        HaMeR crop coords → full image coords.
        Accepts torch or numpy inputs; returns (21,2) np.float32.
        """
        import numpy as np
        kp = self._to_numpy(kp2d).astype(np.float32).copy()  # (21,2)
        cx = self._ensure_float(cx)
        cy = self._ensure_float(cy)
        bs = self._ensure_float(box_size)
        sr = int(self._ensure_float(is_right))               # 0=left, 1=right

        # flip x for right-hand canonicalization (as in your pipeline)
        kp[:, 0] = kp[:, 0] * (2 * sr - 1)
        kp = kp * bs
        kp[:, 0] += cx
        kp[:, 1] += cy
        return kp

    def _frame_hamer_to_lr(self, img_name):
        """
        Route detections by flag (never by index).
        Returns: (L_kp21x2, R_kp21x2, L_scale, R_scale)
        NaNs if a side is missing.
        """
        import numpy as np
        if img_name not in self.hamer:
            nan21 = np.full((21, 2), np.nan, np.float32)
            return nan21, nan21, np.nan, np.nan

        entry     = self.hamer[img_name]
        kp2d_all  = self._to_numpy(entry[0]['pred_keypoints_2d'])  # [N,21,2]
        centers   = self._to_numpy(entry[1])                       # [N,2]
        sizes     = self._to_numpy(entry[2])                       # [N]
        is_rights = self._to_numpy(entry[3])                       # [N] 0/1

        L = np.full((21, 2), np.nan, np.float32); Ls = np.nan
        R = np.full((21, 2), np.nan, np.float32); Rs = np.nan

        N = len(is_rights)
        for i in range(N):
            kp_full = self._unnorm_hand_kp2d(
                kp2d_all[i],
                centers[i][0], centers[i][1],
                sizes[i],
                int(is_rights[i])
            )
            if int(is_rights[i]) == 1:
                R, Rs = kp_full, self._ensure_float(sizes[i])
            else:
                L, Ls = kp_full, self._ensure_float(sizes[i])

        return L, R, Ls, Rs


    def _compute_motion_scores_for_clip(self, tie_ratio=1.2):
 
        LEFT_WRIST_IDX  = 9
        RIGHT_WRIST_IDX = 10

        L_wrists, R_wrists = [], []

        for p in self.img_paths:
            index_key = os.path.join(p.split('/')[-2], p.split('/')[-1])
            kd = self.sapiens[index_key]
            kps  = np.array(kd[0], dtype=np.float32)   # [1, J, 2]
            conf = np.array(kd[1], dtype=np.float32)   # [1, J]

            kps = kps[0]   # [J, 2]
            conf = conf[0] # [J]

            lw = kps[LEFT_WRIST_IDX]  if conf[LEFT_WRIST_IDX]  > 0.3 else np.full(2, np.nan)
            rw = kps[RIGHT_WRIST_IDX] if conf[RIGHT_WRIST_IDX] > 0.3 else np.full(2, np.nan)

            L_wrists.append(lw)
            R_wrists.append(rw)

        L_wrists = np.stack(L_wrists)  # [T, 2]
        R_wrists = np.stack(R_wrists)  # [T, 2]

        def wrist_score(seq):
            diffs = seq[1:] - seq[:-1]          # [T-1, 2]
            v = np.linalg.norm(diffs, axis=-1)  # [T-1]
            return float(np.nanmean(v)) if np.any(np.isfinite(v)) else 0.0

        sL = wrist_score(L_wrists)
        sR = wrist_score(R_wrists)

        eps = 1e-8
        if sL < eps and sR < eps:
            active = 'ambiguous'
        elif not np.isfinite(sL) or sL < eps:
            active = 'right'
        elif not np.isfinite(sR) or sR < eps:
            active = 'left'
        else:
            ratio = (max(sL, sR) + eps) / (min(sL, sR) + eps)
            if ratio < tie_ratio:
                active = 'ambiguous'
            else:
                active = 'left' if sL > sR else 'right'

        T = len(self.img_paths)
        if active == 'left':
            wL = np.full(T, 1.0,  np.float32)
            wR = np.full(T, 0.05, np.float32)
        elif active == 'right':
            wL = np.full(T, 0.05, np.float32)
            wR = np.full(T, 1.0,  np.float32)
        elif active == 'both':
            wL = np.full(T, 1.0,  np.float32)
            wR = np.full(T, 1.0,  np.float32)
        else:
            wL = np.full(T, 0.5,  np.float32)
            wR = np.full(T, 0.5,  np.float32)

        return active, wL, wR

    def get_model2data(self):
        return smpl_to_openpose(self.model_type, use_hands=self.use_hands,
                                use_face=self.use_face,
                                use_face_contour=self.use_face_contour,
                                openpose_format=self.openpose_format)

    def get_left_shoulder(self):
        return 2

    def get_right_shoulder(self):
        return 5

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        # optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
        #                         self.use_face * 51 +
        #                         17 * self.use_face_contour,
        #                         dtype=np.float32)
        optim_weights = np.ones(133, dtype=np.float32)
        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        return self.read_item(img_path)

    def read_item(self, img_path):
        img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
        img_fn, _ = osp.splitext(osp.split(img_path)[1])

        label = ''

        index_key = os.path.join(img_path.split('/')[-2], img_path.split('/')[-1])
        keypoints_dict = self.sapiens[index_key]
        kps = np.array(keypoints_dict[0]).astype(np.float32)
        confidence = np.array(keypoints_dict[1]).astype(np.float32)
        keypoints = np.concatenate((kps, confidence[:, :, None]), axis=-1)
        kp2d_all = self.hamer[img_path.split('/')[-1]][0]['pred_keypoints_2d']
        box_center_all = self.hamer[img_path.split('/')[-1]][1]
        box_size_all = self.hamer[img_path.split('/')[-1]][2]
        is_right_all = self.hamer[img_path.split('/')[-1]][3]

        if self.indp_sign_class != "0":
            for i in range(2):
                kp2d = kp2d_all[i]
                cx, cy = box_center_all[i]
                box_size = box_size_all[i]
                is_right = is_right_all[i]
                # unnormalize to crop coords
                kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
                kp2d = box_size * (kp2d)
                kp2d[:, 0] += cx
                kp2d[:, 1] += cy
                if is_right == 0:
                    keypoints[:, 91:112, :2] = kp2d.cpu().numpy()
                    keypoints[:, 91:112, 2] = 1
                else:
                    keypoints[:, 112:, :2] = kp2d.cpu().numpy()
                    keypoints[:, 112:, 2] = 1
            
            cur_hand = self.hamer[img_path.split('/')[-1]][0]
            cur_left_hand = np.concatenate(
                [cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][0].cpu().numpy()[i])[0] for i in
                range(15)]).squeeze()
            cur_right_hand = np.concatenate(
                [cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][1].cpu().numpy()[i])[0] for i in
                range(15)]).squeeze()
            cur_left_hand = cur_left_hand.reshape(15, 3)[:]
            cur_left_hand[:, 1::3] *= -1
            cur_left_hand[:, 2::3] *= -1
            cur_left_hand = cur_left_hand.reshape(-1)

            cur_hand_3d = cur_hand['pred_keypoints_3d']
            cur_hand_3d = cur_hand_3d.cpu().numpy()
            cam_t = self.hamer[img_path.split('/')[-1]][4]
            cur_hand_3d[0:1, :, 0] = cur_hand_3d[0:1, :, 0] * (-1.)
            cur_hand_3d[0:1] = cur_hand_3d[0:1] + cam_t[0:1, None]
            cur_hand_3d[1:2] = cur_hand_3d[1:2] + cam_t[1:2, None]
            cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))
            base = img_path.split('/')[-1][:-4]
            smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
            with open(smplx_param_path, 'rb') as f:
                smplx_param = pickle.load(f)

            smplx_param['left_hand_pose'] = cur_left_hand
            smplx_param['right_hand_pose'] = cur_right_hand
            label = 'both_hands'

        else:

            if self.one_hand_is_right:

                if len(is_right_all)==1 and is_right_all[0] != 0:

                    kp2d = kp2d_all[0]
                    cx, cy = box_center_all[0]
                    box_size = box_size_all[0]
                    is_right = is_right_all[0]
                    kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
                    kp2d = box_size * (kp2d)
                    kp2d[:, 0] += cx
                    kp2d[:, 1] += cy
                    keypoints[:, 112:, :2] = kp2d.cpu().numpy()
                    keypoints[:, 112:, 2] = 1
                    keypoints[:, 91:112, 2] = 0

                    cur_hand = self.hamer[img_path.split('/')[-1]][0]
                    cur_right_hand = np.concatenate([cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][0].cpu().numpy()[i])[0] for i in range(15)]).squeeze()
                    base = img_path.split('/')[-1][:-4]
                    
                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)

                    smplx_param['right_hand_pose'] = cur_right_hand
                    
                    cur_hand_3d_hamer = cur_hand['pred_keypoints_3d'].cpu().numpy() 
                    cur_hand_3d = np.zeros((1, 21, 3), dtype=cur_hand_3d_hamer.dtype)

                    cam_t = self.hamer[img_path.split('/')[-1]][4]                          
                    cur_hand_3d[0] = cur_hand_3d_hamer[0] + cam_t[0]  
                    
                    cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))
                    label = 'right_hand'     

                    self.prev_2D_keypoints = kp2d
                    self.prev_3D_keypoints = cur_hand_3d
                    self.prev_hand_rotations = cur_right_hand 

                elif len(is_right_all) > 1:
                    ir = self.hamer[img_path.split('/')[-1]][3]
                    ir = ir.detach().cpu().numpy() if torch.is_tensor(ir) else np.asarray(ir)
                    idx_r = int(np.where(ir == 1)[0][0]) if np.any(ir == 1) else 0  # fallback 0

                    kp2d = kp2d_all[idx_r]
                    cx, cy = box_center_all[idx_r]
                    box_size = box_size_all[idx_r]
                    is_right = ir[idx_r]

                    kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
                    kp2d = box_size * (kp2d)
                    kp2d[:, 0] += cx
                    kp2d[:, 1] += cy
                    keypoints[:, 112:, :2] = kp2d.cpu().numpy()
                    keypoints[:, 112:, 2]  = 1
                    keypoints[:, 91:112, 2] = 0

                    cur_hand = self.hamer[img_path.split('/')[-1]][0]

                    cur_hand = self.hamer[img_path.split('/')[-1]][0]
                    cur_right_hand = np.concatenate([cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][1].cpu().numpy()[i])[0] for i in range(15)]).squeeze()
                    base = img_path.split('/')[-1][:-4]

                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)

                    smplx_param['right_hand_pose'] = cur_right_hand

                    cur_hand_3d_hamer = cur_hand['pred_keypoints_3d'].cpu().numpy() 
                    cur_hand_3d = np.zeros((1, 21, 3), dtype=cur_hand_3d_hamer.dtype)

                    cam_t = self.hamer[img_path.split('/')[-1]][4]                          
                    cur_hand_3d[0] = cur_hand_3d_hamer[1] + cam_t[1]  

                    cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))
                    label = 'right_hand'     

                    self.prev_2D_keypoints = kp2d
                    self.prev_3D_keypoints = cur_hand_3d
                    self.prev_hand_rotations = cur_right_hand 


                else:

                    base = img_path.split('/')[-1][:-4]
                    
                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)

                    smplx_param['right_hand_pose'] = self.prev_hand_rotations
                    cur_hand_3d = self.prev_3D_keypoints
                    keypoints[:, 112:, :2] = self.prev_2D_keypoints.cpu().numpy()
                    keypoints[:, 112:, 2] = 1
                    keypoints[:, 91:112, 2] = 0

                    cur_right_hand = self.prev_hand_rotations

                    label = 'right_hand'

            else:
                if len(is_right_all)==1 and is_right_all[0] == 0:
                    kp2d = kp2d_all[0]
                    cx, cy = box_center_all[0]
                    box_size = box_size_all[0]
                    is_right = int(self.one_hand_is_right)
                    kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
                    kp2d = box_size * (kp2d)
                    kp2d[:, 0] += cx
                    kp2d[:, 1] += cy
                    keypoints[:, 91:112, :2] = kp2d.cpu().numpy()
                    keypoints[:, 91:112, 2] = 1
                    keypoints[:, 112:, 2] = 0

                    cur_hand = self.hamer[img_path.split('/')[-1]][0]
                    cur_left_hand = np.concatenate([cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][0].cpu().numpy()[i])[0] for i in range(15)]).squeeze()
                    cur_left_hand = cur_left_hand.reshape(15, 3)[:]
                    cur_left_hand[:, 1::3] *= -1
                    cur_left_hand[:, 2::3] *= -1
                    cur_left_hand = cur_left_hand.reshape(-1)

                    base = img_path.split('/')[-1][:-4]
                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)

                    smplx_param['left_hand_pose'] = cur_left_hand

                    cur_hand_3d_hamer = cur_hand['pred_keypoints_3d'].cpu().numpy()  # → (1, 21, 3)
                    cur_hand_3d = np.zeros((1, 21, 3), dtype=cur_hand_3d_hamer.dtype)

                    cur_hand_3d[0] = cur_hand_3d_hamer[0]

                    cam_t = self.hamer[img_path.split('/')[-1]][4]
                    cur_hand_3d[0, :, 0] *= -1.0            # flip X
                    cur_hand_3d[0] += cam_t[0]             # add translation to left‐hand slice

                    cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))

                    self.prev_2D_keypoints = kp2d
                    self.prev_3D_keypoints = cur_hand_3d
                    self.prev_hand_rotations = cur_left_hand 

                    label = 'left_hand'

                elif len(is_right_all) > 1:
                    kp2d = kp2d_all[0]
                    cx, cy = box_center_all[0]
                    box_size = box_size_all[0]
                    is_right = int(self.one_hand_is_right)
                    kp2d[:, 0] = kp2d[:, 0] * (2 * is_right - 1)
                    kp2d = box_size * (kp2d)
                    kp2d[:, 0] += cx
                    kp2d[:, 1] += cy
                    keypoints[:, 91:112, :2] = kp2d.cpu().numpy()
                    keypoints[:, 91:112, 2] = 1
                    keypoints[:, 112:, 2] = 0

                    cur_hand = self.hamer[img_path.split('/')[-1]][0]
                    cur_left_hand = np.concatenate([cv2.Rodrigues(cur_hand['pred_mano_params']['hand_pose'][0].cpu().numpy()[i])[0] for i in range(15)]).squeeze()
                    cur_left_hand = cur_left_hand.reshape(15, 3)[:]
                    cur_left_hand[:, 1::3] *= -1
                    cur_left_hand[:, 2::3] *= -1
                    cur_left_hand = cur_left_hand.reshape(-1)

                    base = img_path.split('/')[-1][:-4]
                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)

                    smplx_param['left_hand_pose'] = cur_left_hand

                    cur_hand_3d_hamer = cur_hand['pred_keypoints_3d'].cpu().numpy()  # → (1, 21, 3)
                    cur_hand_3d = np.zeros((1, 21, 3), dtype=cur_hand_3d_hamer.dtype)

                    cur_hand_3d[0] = cur_hand_3d_hamer[0]

                    cam_t = self.hamer[img_path.split('/')[-1]][4]
                    cur_hand_3d[0, :, 0] *= -1.0            # flip X
                    cur_hand_3d[0] += cam_t[0]             # add translation to left‐hand slice

                    cur_hand_3d = torch.from_numpy(cur_hand_3d.reshape(-1, 3))

                    self.prev_2D_keypoints = kp2d
                    self.prev_3D_keypoints = cur_hand_3d
                    self.prev_hand_rotations = cur_left_hand 

                    label = 'left_hand'
        
                else:

                    base = img_path.split('/')[-1][:-4]
                    
                    smplx_param_path = os.path.join(self.data_folder, 'smplerx/smplx', f'{base}.pkl')
                    with open(smplx_param_path, 'rb') as f:
                        smplx_param = pickle.load(f)
                    
                    smplx_param['left_hand_pose'] = self.prev_hand_rotations
                    cur_hand_3d = self.prev_3D_keypoints

                    keypoints[:, 91:112, :2] = self.prev_2D_keypoints.cpu().numpy()
                    keypoints[:, 91:112, 2] = 1
                    keypoints[:, 112:, 2] = 0

                    cur_left_hand = self.prev_hand_rotations

                    label = 'left_hand'



    
                
        
        
        smplx_param['betas'] = self.avg_shape

        cur_cam_param = np.zeros((3,3))
        cur_cam_param[0][0] = smplx_param['focal'][0]
        cur_cam_param[1][1] = smplx_param['focal'][1]
        cur_cam_param[0][2] = smplx_param['princpt'][0]
        cur_cam_param[1][2] = smplx_param['princpt'][1]
        cur_cam_param[2][2] = 1.0



        if self.indp_sign_class != "0":
            output_dict = {'fn': img_fn,
                        'img_path': img_path,
                        'cam_param': cur_cam_param,
                        'smplx_param': smplx_param,
                        'pGT_lhand': cur_left_hand,
                        'pGT_rhand': cur_right_hand,
                        'p3DGT_hand': cur_hand_3d,
                        'label': label,
                        'keypoints': keypoints, 'img': img}
        else:
            if label == 'left_hand':
                output_dict = {'fn': img_fn,
                        'img_path': img_path,
                        'cam_param': cur_cam_param,
                        'smplx_param': smplx_param,
                        'pGT_lhand': cur_left_hand,
                        'p3DGT_hand': cur_hand_3d,
                        'label': label,
                        'keypoints': keypoints, 'img': img}
            elif label == 'right_hand':
                output_dict = {'fn': img_fn,
                        'img_path': img_path,
                        'cam_param': cur_cam_param,
                        'smplx_param': smplx_param,
                        'pGT_rhand': cur_right_hand,
                        'p3DGT_hand': cur_hand_3d,
                        'label': label,
                        'keypoints': keypoints, 'img': img}


        return output_dict

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= len(self.img_paths):
            raise StopIteration

        img_path = self.img_paths[self.cnt]
        self.cnt += 1

        return self.read_item(img_path)
