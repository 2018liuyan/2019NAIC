import sys
import os
from inference.BasicModelWarp import Inferrer
from models.archs.EDVR_arch import EDVR
import torch
import numpy as np

import glob
import tqdm

device = 'cuda'

model_args = {
    'nf': 64,
    'nframes': 5,
    'groups': 8,
    'front_RBs': 5,
    'back_RBs': 10,
}

model = EDVR(**model_args)
model.eval()
pth_path = "../model/70000_G.pth"

inferer = Inferrer(device)
inferer.set_model(model)
inferer.load_network(inferer.model, pth_path)
inferer.set_scale(4)
inferer.set_multi_frame_cnt(model_args['nframes'])
inferer.flip_inference = True

video_dir = "../video/SDR_540p"
save_dir = "../video/SDR_540p_srx4_result"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
print("{} videos wait for process...".format(len(video_files)))
for idx, video_path in enumerate(tqdm.tqdm(video_files)):
    base_name = os.path.basename(video_path)
    base_pre = os.path.splitext(base_name)[0]
    save_path = os.path.join(save_dir, base_name)
    im_save_dir = os.path.join(save_dir, "image", base_pre)
    if not os.path.exists(im_save_dir):
        os.makedirs(im_save_dir)
    print(save_path)
    print(im_save_dir)
    inferer.infer_video(video_path,
                        patch_size=None,
                        over_lap=None,
                        save_dir=im_save_dir,
                        save_path=save_path,
                        crf=10,
                        compress_from_folder=True
                        )