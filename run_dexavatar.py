import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='')
parser.add_argument('--output_path',    type=str, default='')
parser.add_argument('--fitting_experiment', type=str, default='')
args = parser.parse_args()

inp_img_folder    = args.input_img_folder
base_output_dir = args.output_path

sub_folder_list = os.listdir(inp_img_folder)

sub_folder_list.sort()

for sub_folder in sub_folder_list:
    input_folder = os.path.join(inp_img_folder, sub_folder)
    out_folder   = os.path.join(base_output_dir, sub_folder)

    os.makedirs(out_folder, exist_ok=True)

    cmd = (
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={args.fitting_experiment} "
        f"bash Full_running_command.sh"
    )
    print("Running:", cmd)
    os.system(cmd)
