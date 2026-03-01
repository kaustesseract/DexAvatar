pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
conda install cuda-toolkit=12.1 -c nvidia/label/cuda-12.1.0
conda install -c conda-forge chumpy
pip install numpy==1.26.3 --no-cache-dir
pip install -v -e hamer/third-party/ViTPose --no-cache-dir
pip install ./torch-mesh-isect --no-build-isolation --no-cache-dir
pip install human-body-prior --no-cache-dir
pip install smplx onnxruntime plyfile lpips==0.1.4 configargparse --no-cache-dir
pip install pytorch-lightning==1.9.0 pyrender timm einops webdataset --no-cache-dir
python -m pip install mmcv==1.3.9 --no-build-isolation --no-cache-dir
pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation --no-cache-dir
pip install -r requirements.txt --no-cache-dir
pip install numpy==1.26.3 --no-cache-dir
pip install ninja --no-cache-dir