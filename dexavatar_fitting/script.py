import os
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--out_path', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--split_num', type=int, default=0)
parser.add_argument('--person_id', type=str, default='')
args = parser.parse_args()

gpu_idx = [args.gpu_id]
split_num = args.split_num
person_list = [args.person_id]
processes = []
for i in range(len(person_list)):
    for j in range(1):
        command = 'CUDA_VISIBLE_DEVICES={} python smplifyx/main.py --config cfg_files/fit_smplx_vposer_x.yaml ' \
                  '--data_folder {} ' \
                  '--output_folder {}  ' \
                  '--img_folder {}  ' \
                  '--model_folder ../SMPLer-X/common/utils/human_model_files ' \
                  '--part_segm_fn assets/smplx_parts_segm.pkl --visualize False --split_num {} --cur_num {} ' \
            .format(str(gpu_idx[0]), args.out_path, os.path.join(args.out_path, 'smplifyx'), args.path, split_num, j+0)
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
output = [p.wait() for p in processes]

