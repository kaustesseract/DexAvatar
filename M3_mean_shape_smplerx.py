import os
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
args = parser.parse_args()

root_path = args.input_path
out_path = args.output_path
path = os.path.join(out_path, 'smplerx/smplx')
smplx_list = os.listdir(path)
beta_list = []
for smplx_name in smplx_list:
    smplx_path = os.path.join(path, smplx_name)
    with open(smplx_path, 'rb') as f:
        smplx_param = pickle.load(f)
    beta_list.append(smplx_param['betas'])
beta_avg = np.stack(beta_list).mean(axis=0)
out_path = os.path.join(out_path.replace('SignLanguage_S1', ''), 'mean_shape_smplx.npy')
np.save(out_path, beta_avg)
