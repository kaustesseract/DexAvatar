#!/bin/bash

conda_env="smpler_x"
file="$(conda info --base)/envs/$conda_env/lib/python3.8/site-packages/torchgeometry/core/conversions.py"
if [ ! -f "$file" ]; then
    echo "Fail：$file does not exist!"
    exit 1
fi
sed -i -E '/mask_c[0-9]/s/\(1 - /~(/g' "$file"
echo "Complete: $file"