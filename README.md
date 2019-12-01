# 2019NAIC
2019 naic challege code

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  ```
  cd ./codes/models/archs/dcn
  python setup.py develop
  ```
- Python packages: `pip install numpy opencv-python lmdb pyyaml tqdm tensorboardX scikit-video`
- Other packages: FFMPEG
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`


## Dataset Preparation
运行codes/data_scripts下的 video_to_img_lmdb.py 生成训练用的png和lmdb文件.

## Training
在codes目录下，运行 `python train.py -opt options/train/ly_20191123_edvr_continue_ft.yml`

## Inference for NAIC-2019
1. 修改codes/naic_inference.py 中的video_dir为待处理视频所在的目录
2. 修改codes/naic_inference.py 中的save_dir为你期望结果保存目录
3. 在codes目录下，运行 `python naic_inference.py`
