import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--split_num', type=int, default=0)
parser.add_argument('--output_path', type=str, default='')
args = parser.parse_args()

gpu_idx = [args.gpu_id]
split_num = args.split_num
processes = []


for j in range(1):
    command = 'CUDA_VISIBLE_DEVICES={} python inference.py --num_gpus 1 ' \
              '--exp_name output --pretrained_model smpler_x_h32 --agora_benchmark agora_model ' \
              '--img_path {}/images ' \
              '--output_folder {}/smplerx --show_verts --show_bbox --save_mesh --split_num {} --cur_num {} ' \
        .format(str(gpu_idx[0]), args.path, args.output_path, split_num, j+0)
    print(command)
    process = subprocess.Popen(command, shell=True)
    processes.append(process)
output = [p.wait() for p in processes]

