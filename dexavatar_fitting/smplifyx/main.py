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
import json
import time
import yaml
import torch

import smplx
from rewrite_body_model import SMPLX

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior
import math

torch.backends.cudnn.enabled = False
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    mapping_path = args['sign_class']

    class_sign = {}

    with open(mapping_path, 'r') as f:
        for line in f:
            tokens = line.split(' ')
            tokens = list(map(lambda x: x.strip('\n'), tokens))
            class_sign[tokens[0]] = tokens[1]
    
    class_sign = dict(sorted(class_sign.items(), key=lambda x: x[0]))

    folder_name = args.get("img_folder").split('/')[-1]

    img_path = args.get('img_folder')

    img_path = os.path.join(img_path, 'images')

    indp_sign_class = class_sign[folder_name]

    #img_folder = args.pop('img_folder', 'images')
    
    with open(args.get('sign_segment'), 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    
    dataset_obj = create_dataset(indp_sign_segment=data[folder_name], img_path=img_path, indp_sign_class=indp_sign_class, **args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        # neutral_model = smplx.create(gender='neutral', **model_params)
        neutral_model = SMPLX(model_path='../SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl', ext='pkl',
                     use_face_contour=True, flat_hand_mean=True, use_pca=False,
                     num_betas=10, num_expression_coeffs=10)
    female_model = smplx.create(gender='female', **model_params)

    # Create the camera object
    
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    joints_temp = torch.zeros([1,21,3])
    
    joints_temp=joints_temp.to(device=device)

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    total_number = len(dataset_obj)
    total_split_num = args.get('split_num')
    cur_num = args.get('cur_num')
    chunk_num = math.ceil(total_number/total_split_num)

    #print('Total number of images: {}'.format(total_number))

    for idx, data in enumerate(dataset_obj):
        if idx >= chunk_num * cur_num and idx < chunk_num * (cur_num+1):
            pass
        else:
            continue
        img = data['img']
        fn = data['fn']
        cam_param = data['cam_param'] # camera parameters probably got from SMPLer-X
        cam_param = torch.from_numpy(cam_param).to(device=device)
        init_smplx_param = data['smplx_param'] # SMPLer-X parameters (body, hand, face .. )
        p3DGT_hand = data['p3DGT_hand'] # Hamer 3D pose parameters

        keypoints = data['keypoints']
        print('Processing: {}'.format(data['img_path']))

        curr_result_folder = osp.join(result_folder)
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(mesh_folder)
        if not osp.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)
        for person_id in range(keypoints.shape[0]):
            if person_id >= max_persons and max_persons > 0:
                continue

            curr_result_fn = osp.join(curr_result_folder,
                                      '{}.pkl'.format(fn))
            curr_mesh_fn = osp.join(curr_mesh_folder,
                                    '{}.obj'.format(fn))

            curr_img_folder = osp.join(output_folder, 'images')
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][person_id]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][person_id]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            # if data['label'] == 'left_hand':
            #     body_model.right_hand_pose.requires_grad = False

            # if data['label'] == 'right_hand' or indp_signs == "0":
            #     body_model.left_hand_pose.requires_grad = False

            #print(data['keypoints'][:,91:])
            #print(data['keypoints'].shape)

            out_img_fn = osp.join(curr_img_folder, fn+'.png')
            if os.path.exists(out_img_fn):
                continue

            joints_temp_re = fit_single_frame(img, keypoints[[person_id]], data['img_path'],
                             body_model=body_model,
                             idx=idx,
                             camera=cam_param,
                             init_smplx_param=init_smplx_param,
                             p3DGT_hand=p3DGT_hand,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             joints_temp=joints_temp,
                             angle_prior=angle_prior,
                             indp_sign_class=indp_sign_class,
                             hand_label=data['label'],
                             **args)
            
            joints_temp = joints_temp_re.clone()

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    
    # import pdb
    # pdb.set_trace()
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
