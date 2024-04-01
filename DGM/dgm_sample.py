import os
import torch
import argparse
import threading

import numpy as np

from denoising_diffusion_models.denoising_diffusion_pytorch import Trainer
from denoising_diffusion_models.classifier_free_guidance import Unet, GaussianDiffusion

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, default='None')
parser.add_argument('--gpu_nums', type=int, default=0)
parser.add_argument('--s_step', type=int, default=0)
parser.add_argument('--part', type=int, default=0)
parser.add_argument('--bs', type=int, default=80)
parser.add_argument('--exp', type=str)
parser.add_argument('-i', type=int, default=0)
args = parser.parse_args()


num_classes = 1


def parallel_evaluation(rank, idx, part, bs):
    trainList = []

    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=6, num_classes=num_classes).cuda(rank)

    diffusion = GaussianDiffusion(
        model,
        image_size=256,
        timesteps=1000,  # number of steps
        sampling_timesteps=args.s_step,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l1',  # L1 or L2
        # p2_loss_weight_gamma=1.,
        objective='pred_x0',
    ).cuda(rank)

    trainer = Trainer(
        diffusion,
        'DGM_Conditions',
        train_batch_size=bs,
        train_lr=1e-4,
        train_num_steps=200000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        results_folder="results",
        save_and_sample_every=2000,
        num_samples=4,
        augment_horizontal_flip=False,
        num_worker=0,
        total_data_slice_idx=args.gpu_nums,
        data_slice_idx=args.i,
        shuffle=False,
        isGenerate=True,
    )

    trainer.load(args.c)

    while True:
        try:
            ret = trainer.sample(idx, rank, step=len(trainList))
            trainList.append(ret)
            torch.cuda.empty_cache()

            print(f"length of trainList {len(trainList)}")
            if not os.path.exists(f'traindata/{args.exp}/dataset/'):
                os.makedirs(f'traindata/{args.exp}/dataset/')

            # save training data every n steps
            if len(trainList) % 2 == 0:
                np.save(f'traindata/{args.exp}/dataset/idx_{idx}_rank_{rank}_part_{part}_dm_cahomo_{len(trainList) * bs /1000}k.npy', trainList)

                trainList.clear()
                part += 1

        except Exception as e:
            raise e


if __name__ == "__main__":
    threads = []
    idx = args.i
    part = args.part
    bs = args.bs

    # for i in range(torch.cuda.device_count()):
    for i in range(1):
        t = threading.Thread(target=parallel_evaluation, args=(
            i,
            idx,
            part,
            bs,
        ))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()
