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
- Python packages: `pip install numpy opencv-python lmdb pyyaml`
- TensorBoard:
  - PyTorch >= 1.1: `pip install tb-nightly future`


## Dataset Preparation
We use datasets in LDMB format for faster IO speed. Run script video_to_img_lmdb.py, generate .png and lmdb files.

## Training
run python train -opt options/train/ly_20191123_edvr_continue_ft.yml
