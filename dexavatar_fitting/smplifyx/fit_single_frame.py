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


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict
from test_bposer import load_signbposer
import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from test_hposer import load_hposer3d
import matplotlib.pyplot as plt


def fit_single_frame(img,
                     keypoints,
                     img_path,
                     body_model,
                     idx,
                     camera,
                     init_smplx_param,
                     p3DGT_hand,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     joints_temp=None,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     data_3d_weights=None,
                     data_init_core_weights=None,
                     data_init_noncore_weights=None,
                     data_init_lhand_weights=None,
                     data_init_rhand_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     indp_sign_class=None,
                     hand_label=None,
                     body_joints_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     body_biomechanics_loss_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     coll_loss_weights=None,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5
    if data_3d_weights is None:
        data_3d_weights = [0, ] * 5
    if data_init_core_weights is None:
        data_init_core_weights = [0, ] * 5
    if data_init_noncore_weights is None:
        data_init_noncore_weights = [0, ] * 5
    if data_init_lhand_weights is None:
        data_init_lhand_weights = [0, ] * 5
    if data_init_rhand_weights is None:
        data_init_rhand_weights = [0, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if body_joints_weights is None:
            body_joints_weights = [1.0, 1.0, 1.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(body_joints_weights) ==
                    len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg
    
    if body_biomechanics_loss_weights is None:
        body_biomechanics_loss_weights = [0, 0, 0, 0]
    msg = ('Number of Body pose prior weights = {} does not match the' +
            ' number of Expression prior weights = {}')
    assert (len(body_biomechanics_loss_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(body_pose_prior_weights),
                len(body_biomechanics_loss_weights))

    use_signbposer = kwargs.get('use_signbposer', True)
    
    signbposer, pose_embedding = [None, ] * 2
    
    if use_signbposer:
        pose_embedding = torch.zeros([batch_size, 33],
                                     dtype=dtype, device=device,
                                     requires_grad=True) # the size is 32 because the latent representation of vposer is 32

        signbposer, _ = load_signbposer(kwargs.get('signbposer_dir')) # loading the vposer
        signbposer = signbposer.to(device=device)
        signbposer.eval()

    use_hposer3d = kwargs.get('use_hposer3d', True)
    hposer3d, rhand_embedding3d, lhand_embedding3d = [None, ] * 3


    if use_hposer3d:

        lhand_embedding3d = torch.zeros([batch_size, 23], dtype=dtype, device=device, requires_grad=True)

        rhand_embedding3d = torch.zeros([batch_size, 23], dtype=dtype, device=device, requires_grad=True)
        
        hposer3d, _ = load_hposer3d(kwargs.get('signhposer_dir'))
        hposer3d = hposer3d.to(device=device)
        hposer3d.eval()

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)
    
    p3DGT_hand = p3DGT_hand.to(device=device, dtype=dtype)
    psmplx_bodyGT = torch.from_numpy(init_smplx_param['body_pose'][None]).to(device=device, dtype=dtype)
    psmplx_lhandGT = torch.from_numpy(init_smplx_param['left_hand_pose'][None]).to(device=device, dtype=dtype)
    psmplx_rhandGT = torch.from_numpy(init_smplx_param['right_hand_pose'][None]).to(device=device, dtype=dtype)

    if idx==0:
        joints_temp=torch.from_numpy(init_smplx_param['body_pose'][None]).reshape(1,21,3).to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None
    if interpenetration:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'

        search_tree = BVH(max_collisions=max_collisions)

        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=df_cone_height, point2plane=point2plane,
                vectorized=True, penalize_outside=penalize_outside)

        if part_segm_fn:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(part_segm_fn)
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                             encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=ign_part_pairs).to(device=device)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'data_3d_weight': data_3d_weights,
                        'data_init_core_weight': data_init_core_weights,
                        'data_init_noncore_weight': data_init_noncore_weights,
                        'data_init_lhand_weight': data_init_lhand_weights,
                        'data_init_rhand_weight': data_init_rhand_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'body_weight': body_joints_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if body_biomechanics_loss_weights:
        opt_weights_dict['body_biomechanics_loss_weights'] = body_biomechanics_loss_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = kwargs.get('body_tri_idxs')
    init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                use_signbposer=use_signbposer, signbposer=signbposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               signbposer=signbposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        img = torch.tensor(img, dtype=dtype)

        H, W, _ = img.shape

        data_weight = 1000 / H # This normalizes the loss across images of different sizes, so the magnitude of the 2D joint errors is similar no matter the resolution

        orientations = [init_smplx_param['global_orient'][None]]
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            if indp_sign_class != "0":
                new_params = defaultdict(global_orient=orient,
                                        betas=init_smplx_param['betas'][None],
                                        body_pose=init_smplx_param['body_pose'][None],
                                        body_pose_fore=init_smplx_param['body_pose'][:15*3][None],
                                        body_pose_op=init_smplx_param['body_pose'][15*3:][None],
                                        jaw_pose=init_smplx_param['jaw_pose'][None],
                                        left_hand_pose=init_smplx_param['left_hand_pose'][None],
                                        right_hand_pose=init_smplx_param['right_hand_pose'][None],
                                        leye_pose=init_smplx_param['leye_pose'][None],
                                        reye_pose=init_smplx_param['reye_pose'][None],
                                        expression=init_smplx_param['expression'][None],
                                        transl=init_smplx_param['transl'][None],
                                        ) 
            else:
                if hand_label == 'right_hand':
                    new_params = defaultdict(global_orient=orient,
                                        betas=init_smplx_param['betas'][None],
                                        body_pose=init_smplx_param['body_pose'][None],
                                        body_pose_fore=init_smplx_param['body_pose'][:15*3][None],
                                        body_pose_op=init_smplx_param['body_pose'][15*3:][None],
                                        jaw_pose=init_smplx_param['jaw_pose'][None],
                                        right_hand_pose=init_smplx_param['right_hand_pose'][None],
                                        leye_pose=init_smplx_param['leye_pose'][None],
                                        reye_pose=init_smplx_param['reye_pose'][None],
                                        expression=init_smplx_param['expression'][None],
                                        transl=init_smplx_param['transl'][None],
                                        ) 
                    
                elif hand_label == 'left_hand':
                    new_params = defaultdict(global_orient=orient,
                                        betas=init_smplx_param['betas'][None],
                                        body_pose=init_smplx_param['body_pose'][None],
                                        body_pose_fore=init_smplx_param['body_pose'][:15*3][None],
                                        body_pose_op=init_smplx_param['body_pose'][15*3:][None],
                                        jaw_pose=init_smplx_param['jaw_pose'][None],
                                        left_hand_pose=init_smplx_param['left_hand_pose'][None],
                                        leye_pose=init_smplx_param['leye_pose'][None],
                                        reye_pose=init_smplx_param['reye_pose'][None],
                                        expression=init_smplx_param['expression'][None],
                                        transl=init_smplx_param['transl'][None],
                                        ) 

            # new_params = defaultdict(global_orient=orient,
            #                         betas=init_smplx_param['betas'][None],
            #                         body_pose=init_smplx_param['body_pose'][None],
            #                         body_pose_fore=init_smplx_param['body_pose'][:15*3][None],
            #                         body_pose_op=init_smplx_param['body_pose'][15*3:][None],
            #                         jaw_pose=init_smplx_param['jaw_pose'][None],
            #                         left_hand_pose=init_smplx_param['left_hand_pose'][None],
            #                         right_hand_pose=init_smplx_param['right_hand_pose'][None],
            #                         leye_pose=init_smplx_param['leye_pose'][None],
            #                         reye_pose=init_smplx_param['reye_pose'][None],
            #                         expression=init_smplx_param['expression'][None],
            #                         transl=init_smplx_param['transl'][None],
            #                         ) 


            body_model.reset_params(**new_params) # This is the place where you initialise the body model parameters for the SMPL-X layer
            if use_signbposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)
            
            if use_hposer3d:
                    with torch.no_grad():

                        if indp_sign_class != "0":
                            lhand_embedding3d.fill_(0)
                            rhand_embedding3d.fill_(0)

                        else:
                            if hand_label == 'left_hand':
                                lhand_embedding3d.fill_(0)

                            elif hand_label == 'right_hand':
                                rhand_embedding3d.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                # body_params = list(body_model.parameters())
                # body_params = []
                # for k, v in body_model.named_parameters():
                #     if 'hand' in k and (opt_idx != 0 and opt_idx != 1):
                #         print('hand added')
                #         body_params.append(v)

                    # if (('jaw' in k) or ('expr' in k) or ('eye' in k)) and (opt_idx != 0):
                    #     print(f"{k} added")
                    #     body_params.append(v)


                final_params = []

                if use_signbposer:
                    final_params.append(pose_embedding)

                if use_hposer3d:

                    if indp_sign_class != "0":
                        final_params.append(lhand_embedding3d)
                        final_params.append(rhand_embedding3d)

                    else:
                        if hand_label == 'left_hand':
                            final_params.append(lhand_embedding3d)

                        elif hand_label == 'right_hand':
                            final_params.append(rhand_embedding3d)

                

                #tqdm.write('Stage {:03d} has parameters {:01d}'.format(opt_idx, len(final_params)))

                # FINAL PARAMS LIST ONLY HAS POSE EMBEDDING PARAMETERS FOR STAGE 0 AND 1. FOR STAGE 2 IT IS ADDING HAND PARAMETERS.
                # SO FOR STAGE 0 AND 1, THE LENGTH OF FINAL PARAMS IS 1, AND FOR STAGE 2 IT IS 3 BECAUASE (1 + 2 HANDS)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['data_3d_weight'] = curr_weights['data_3d_weight']
                curr_weights['data_init_core_weight'] = curr_weights['data_init_core_weight']
                curr_weights['data_init_noncore_weight'] = curr_weights['data_init_noncore_weight']
                curr_weights['data_init_lhand_weight'] = curr_weights['data_init_lhand_weight']
                curr_weights['data_init_rhand_weight'] = curr_weights['data_init_rhand_weight']
                curr_weights['body_biomechanics_loss_weights'] = curr_weights['body_biomechanics_loss_weights']
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                
                if indp_sign_class != "0":
                    joint_weights[:, 5:11] = curr_weights['body_weight']
                else:
                    if hand_label == "right_hand":
                        joint_weights[:, 6] = curr_weights['body_weight']
                        joint_weights[:, 8] = curr_weights['body_weight']
                        joint_weights[:, 10] = curr_weights['body_weight']
                        joint_weights[:, 5] = 0
                        joint_weights[:, 7] = 0
                        joint_weights[:, 9] = 0
                    elif hand_label == "left_hand":
                        joint_weights[:, 5] = curr_weights['body_weight']
                        joint_weights[:, 7] = curr_weights['body_weight']
                        joint_weights[:, 9] = curr_weights['body_weight']
                        joint_weights[:, 6] = 0
                        joint_weights[:, 8] = 0
                        joint_weights[:, 10] = 0


                if use_hands:
                    if indp_sign_class != "0":
                        joint_weights[:, 91:] = curr_weights['hand_weight']
                    else:
                        if hand_label == 'right_hand':
                            joint_weights[:, 112:] = curr_weights['hand_weight']
                            joint_weights[:, 91:112] = 0
                        elif hand_label == 'left_hand':
                            joint_weights[:, 91:112] = curr_weights['hand_weight']
                            joint_weights[:, 112:] = 0

                if use_face:
                    joint_weights[:, 23:91] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints, p3DGT_hand=p3DGT_hand, psmplx_bodyGT=psmplx_bodyGT, psmplx_lhandGT=psmplx_lhandGT, psmplx_rhandGT=psmplx_rhandGT,
                    hposer3d=hposer3d, lhand_embedding3d=lhand_embedding3d, rhand_embedding3d=rhand_embedding3d, use_hposer3d=use_hposer3d,
                    joints_conf=joints_conf, indp_sign_class=indp_sign_class, hand_label=hand_label, joints_temp=joints_temp,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_signbposer=use_signbposer, signbposer=signbposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, signbposer=signbposer,
                    use_signbposer=use_signbposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_signbposer:
                result['body_pose'] = signbposer.decode(pose_embedding, output_type='aa').view(1, -1).detach().cpu().numpy()
            result['K'] = camera.cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

    if save_meshes or visualize:
        body_pose = signbposer.decode(
            pose_embedding,
            output_type='aa').view(1, -1) if use_signbposer else None

        model_type = kwargs.get('model_type', 'smpl')
        append_wrists = model_type == 'smpl' and use_signbposer
        if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

        # model_output = body_model(body_pose=body_pose, return_verts=True)
        # vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        if use_hposer3d:

            if indp_sign_class != "0":
                    lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                    rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, right_hand_pose=rhand_pose, left_hand_pose=lhand_pose)
            else:
                if hand_label == 'right_hand':
                    rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, right_hand_pose=rhand_pose)
                
                elif hand_label == 'left_hand':
                    lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                    model_output = body_model(return_verts=True, body_pose=body_pose, left_hand_pose=lhand_pose)


        else:

            model_output = body_model(body_pose=body_pose, return_verts=True)
        
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        
        import trimesh
        import neural_renderer as nr

        _, faces = nr.load_obj('assets/smplx_uv_new.obj')
        faces = faces.cpu().numpy()
        out_mesh = trimesh.Trimesh(vertices, faces)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(mesh_fn)

    if True:
        import pyrender
        COLOR_MESH = [
            (0.5, 0.2, 0.2, 1.0),  # Defalut
            (.7, .5, .5, 1.),  # Pink
            (.7, .7, .6, 1.),  # Neutral
            (.5, .5, .7, 1.),  # Blue
            (.5, .55, .3, 1.),  # capsule
            (.3, .5, .55, 1.),  # Yellow
        ]


        # offscreen version
        fx = camera[0][0].item()
        fy = camera[1][1].item()
        cx = camera[0][2].item()
        cy = camera[1][2].item()
        # Create a scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))
        camera_pose = np.eye(4)
        camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(camera, pose=camera_pose)


        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=COLOR_MESH[3])
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        scene.add(mesh, 'mesh')


        # Get the lights from the viewer
        # light_nodes = monitor.mv.viewer._create_raymond_lights()
        # for node in light_nodes:
        #     scene.add_node(node)
        # Use 3 directional lights
        light_pose = np.eye(4)
        trans = [0, 0, 0]
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)

        r = pyrender.OffscreenRenderer(viewport_width=W,
                                       viewport_height=H,)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        input_img = img.detach().cpu().numpy()
        output_img = (color[:, :, :-1] * valid_mask +
                      (1 - valid_mask) * input_img)

        plt.figure()
        plt.imshow(input_img)
        plt.imshow(color, alpha=0.9)
        plt.axis('off')
        plt.savefig(out_img_fn, bbox_inches='tight', dpi=260)

        joints_temp = model_output.body_pose.reshape(1,21,3).detach()

        return joints_temp
