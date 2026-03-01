#!/bin/bash
#export OUT_PATH=${ROOT_PATH}/smplifyx
CONDA_PATH=$(conda info --base)
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate sapiens_lite