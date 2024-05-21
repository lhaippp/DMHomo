import cv2
import pdb
import torch
import argparse

import numpy as np

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import Trainer
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=0)
args = parser.parse_args()

num_classes = 5

model = Unet(dim=64,
             dim_mults=(1, 2, 4, 8),
             channels=6,
             num_classes=num_classes)

diffusion = GaussianDiffusion(
    model,
    image_size=256,
    timesteps=1000,  # number of steps
    sampling_timesteps=
    32,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1',  # L1 or L2
    # p2_loss_weight_gamma=1.,
    objective='pred_x0',
)

# this setting is test under 8 H100s with each of 80G GPU memory
train_batch_size = 256 // 2
data_num = 450000
epoch = 32
print(f'total steps are: {(data_num * epoch) // train_batch_size}')

trainer = Trainer(
    diffusion,
    'DMHomo',
    train_batch_size=train_batch_size,
    train_lr=1e-4 * 10 / 2,
    train_num_steps=(data_num * epoch) //
    train_batch_size,  # total training steps, loop 10 epoches
    gradient_accumulate_every=1,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    results_folder="results",
    save_and_sample_every=1000,
    num_samples=9,
    augment_horizontal_flip=False)

# trainer.results_folder = "results"
if args.c != 0:
    trainer.load(args.c)

trainer.train()
