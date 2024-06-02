"""Evaluates the model"""

import os
import random
import logging
import argparse
import torchvision

import cv2
import imageio
import torch.optim as optim
import itertools
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from loss.losses import compute_eval_results
from common.manager import Manager

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from model.utils import get_warp_flow
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/Bases_Ours/', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='experiments/Bases_Ours/val_model_best_0.6689.pth',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'experiments/version1/model_latest.pth'
parser.add_argument('-ow', '--only_weights', action='store_true', default=True, help='Only use weights to load or load all train status.')


def make_align_heatmap(ref, inp):
    heat_map = inp.astype(np.float32) - ref.astype(np.float32)
    heat_map = abs(heat_map)
    heat_map -= heat_map.min()
    heat_map /= heat_map.max()
    heat_map = np.clip(heat_map, 0.01, 1)
    heat_map = cv2.applyColorMap((heat_map * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1].astype(np.uint8)
    # heat_map = append_text_information(heat_map, text, ssim_scores)
    return heat_map

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    if manager.accelerator.is_main_process:
        manager.accelerator.print("\n")
        manager.logger.info("eval begin!")

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    # set_seed(42)

    torch.cuda.empty_cache()
    model.eval()

    idx = 0
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")

            for data_batch in manager.dataloaders["val"]:
                # compute model output
                output = model(data_batch, inference=False)

                video_name = data_batch["video_names"]
                # compute all loss on this batch
                err_avg = compute_eval_results(data_batch, output)

                for j in range(len(err_avg)):
                    if video_name[j] in RE:
                        MSE_RE.append(err_avg[j])
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])
                    else:
                        raise Exception(f'video_name {video_name[j]} exception')

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)

            manager.accelerator.wait_for_everyone()

            # results shoud be gatherd from multiple GPUs to single GPU
            MSE_RE_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_RE_avg).cuda(manager.accelerator.process_index)))
            MSE_LT_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LT_avg).cuda(manager.accelerator.process_index)))
            MSE_LL_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LL_avg).cuda(manager.accelerator.process_index)))
            MSE_SF_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_SF_avg).cuda(manager.accelerator.process_index)))
            MSE_LF_avg = torch.mean(manager.accelerator.gather_for_metrics(
                torch.tensor(MSE_LF_avg).cuda(manager.accelerator.process_index)))

            manager.accelerator.wait_for_everyone()

        if manager.accelerator.is_main_process:
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5
            # manager.accelerator.print(f"MSE_avg: {MSE_avg}")

            Metric = {
                "MSE_RE_avg": MSE_RE_avg,
                "MSE_LT_avg": MSE_LT_avg,
                "MSE_LL_avg": MSE_LL_avg,
                "MSE_SF_avg": MSE_SF_avg,
                "MSE_LF_avg": MSE_LF_avg,
                "AVG": MSE_avg
            }
            manager.update_metric_status(metrics=Metric, split="val")

            manager.logger.info("Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val, MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("val", title="val", color="green")

            manager.epoch_val += 1

            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            model.train()
            val_metrics = {'MSE_avg': MSE_avg}
            return val_metrics


def mask_vis(model, manager):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode
    if manager.accelerator.is_main_process:
        manager.accelerator.print("\n")
        manager.logger.info("eval begin!")
    # # save psedo dm-mask
    # mask_fusion_ls = []

    torch.cuda.empty_cache()
    model.eval()
    idx = 0

    if not os.path.exists(f'./visualization/{manager.params.exp_name}'):
        os.mkdir(f'./visualization/{manager.params.exp_name}')

        os.mkdir(f'./visualization/{manager.params.exp_name}/RE')
        os.mkdir(f'./visualization/{manager.params.exp_name}/LT')
        os.mkdir(f'./visualization/{manager.params.exp_name}/LL')
        os.mkdir(f'./visualization/{manager.params.exp_name}/SF')
        os.mkdir(f'./visualization/{manager.params.exp_name}/LF')

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239', '0000030']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']


    re_idx = 0
    lt_idx = 0
    ll_idx = 0
    sf_idx = 0
    lf_idx = 0
 
    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["val"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("val")

            for data_batch in manager.dataloaders["val"]:
                video_name = data_batch['video_names'][0]
                image_name = data_batch['save_name'][0]

                # if not (('00000141_10009' in image_name) or ('00000200_10266' in image_name) or ('00000200_10470' in image_name) or
                #         ('30_10024' in image_name) or ('107_10459' in image_name) or ('107_10763' in image_name) or ('00000200_10418' in image_name)):
                #     print(f'{image_name} is not the target, continue')
                #     continue
                # if not ('00000200_10418' in image_name):
                #     print(f'{image_name} is not the target, continue')
                #     continue

                if video_name in RE:
                    base_dir = 'RE'
                    re_idx += 1
                    if re_idx >= 50:
                        print(f're_idx is {re_idx}')
                        continue
                elif video_name in LT:
                    base_dir = 'LT'
                    lt_idx += 1
                    if lt_idx >= 50:
                        print(f'lt_idx is {lt_idx}')
                        continue
                elif video_name in LL:
                    base_dir = 'LL'
                    ll_idx += 1
                    if ll_idx >= 50:
                        print(f'll_idx is {ll_idx}')
                        continue
                elif video_name in SF:
                    base_dir = 'SF'
                    sf_idx += 1
                    if sf_idx >= 50:
                        print(f'sf_idx is {sf_idx}')
                        continue
                elif video_name in LF:
                    base_dir = 'LF'
                    lf_idx += 1
                    if lf_idx >= 50:
                        print(f'lf_idx is {lf_idx}')
                        continue

                output = model(data_batch)
                rgb_imgs = data_batch["imgs_rgb_full"]
                im1, im2 = rgb_imgs[:, :3], rgb_imgs[:, 3:]
                flow_b = output["flow_b"]
                flow_b = flow_b.permute(0, 3, 1, 2)
                # im2_remap = get_warp_flow(im2, flow_f)
                im1_remap = get_warp_flow(im1, flow_b)

                # mask_fusion = output["mask_fusion"]
                mask_f = output["mask_f"]
                mask_b = output["mask_b"]

                mask_f = (mask_f / torch.max(mask_f * 0.5)).clamp(0, 1)
                mask_b = (mask_b / torch.max(mask_b * 0.5)).clamp(0, 1)

                print_img_1_d = im1.cpu().detach().numpy()[0, ...]
                print_img_2_d = im2.cpu().detach().numpy()[0, ...]
                im1_remap = im1_remap.cpu().detach().numpy()[0, ...]
                print_img_1_d = (np.transpose(print_img_1_d, [1, 2, 0]) * 255).astype(np.uint8)
                print_img_2_d = (np.transpose(print_img_2_d, [1, 2, 0]) * 255).astype(np.uint8)
                im1_remap = (np.transpose(im1_remap, [1, 2, 0]) * 255).astype(np.uint8)

                mask_f = mask_f.cpu().detach().numpy()[0, ...]
                mask_b = mask_b.cpu().detach().numpy()[0, ...]
                mask_f = np.transpose(mask_f, [1, 2, 0])
                mask_b = np.transpose(mask_b, [1, 2, 0])

                # print(f"print_img_1_d {print_img_1_d.shape} | mask_f {mask_f.shape}")

                heatmap_img1 = show_cam_on_image(im1_remap / 255., mask_f)
                heatmap_img2 = show_cam_on_image(print_img_2_d / 255., mask_b)

                torchvision.utils.save_image(im2[:, [2, 1, 0]],
                                             f"./visualization/{manager.params.exp_name}/{base_dir}/{image_name}_img2.jpg",
                                             nrow=1)
                torchvision.utils.save_image(im1[:, [2, 1, 0]],
                                             f"./visualization/{manager.params.exp_name}/{base_dir}/{image_name}_img1.jpg",
                                             nrow=1)
                # torchvision.utils.save_image(im1_remap[:, [2, 1, 0]], f"./visualization/{base_dir}/{image_name}_im1_remap.png', nrow=1)
                # torchvision.utils.save_image(mask_fusion, f'unit_test/dmhomo_generates/{idx}_mask_fusion.png', nrow=1)
                cv2.imwrite(f'./visualization/{manager.params.exp_name}/{base_dir}/{image_name}_mask_1.jpg', heatmap_img1)
                cv2.imwrite(f'./visualization/{manager.params.exp_name}/{base_dir}/{image_name}_mask_2.jpg', heatmap_img2)

                print_img_1_d = cv2.cvtColor(print_img_1_d, cv2.COLOR_BGR2RGB)
                print_img_2_d = cv2.cvtColor(print_img_2_d, cv2.COLOR_BGR2RGB)
                im1_remap = cv2.cvtColor(im1_remap, cv2.COLOR_BGR2RGB)
                heatmap_img1 = Image.fromarray(cv2.cvtColor(heatmap_img1, cv2.COLOR_BGR2RGB))
                heatmap_img2 = Image.fromarray(cv2.cvtColor(heatmap_img2, cv2.COLOR_BGR2RGB))

                tmp_img1 = Image.fromarray(print_img_1_d)
                tmp_img2 = Image.fromarray(print_img_2_d)
                frame1 = Image.fromarray(np.hstack([tmp_img1, im1_remap, heatmap_img1, heatmap_img2]))
                frame2 = Image.fromarray(np.hstack([tmp_img2, tmp_img2, heatmap_img1, heatmap_img2]))
                full_img = [frame1, frame2]
                imageio.mimsave(f'./visualization/{manager.params.exp_name}/{base_dir}/{image_name}.gif', full_img, 'GIF', duration=0.5)

                # input()

                # torchvision.utils.save_image(buf_color_ghost[:, [2, 1, 0]], f'unit_test/dmhomo_generates/color_ghost/{idx}.png', nrow=1)

                # make_align_heatmap(
                #     f'unit_test/dmhomo_generates/{idx}_img2.png',
                #     f'unit_test/dmhomo_generates/{idx}_im1_remap.png',
                #     'dmhomo_generates',
                #     f'{idx}',
                # )

                # make_align_heatmap(
                #     f'unit_test/dmhomo_generates/{idx}_img2.png',
                #     f'unit_test/dmhomo_generates/{idx}_img1.png',
                #     'dmhomo_generates',
                #     f'{idx}_motion12',
                # )

                # idx += 1


def test(model, manager):
    """Test the model with loading checkpoints.

    Args:
        model: (torch.nn.Module) the neural network
        manager: a class instance that contains objects related to train and evaluate.
    """
    # set model to evaluation mode

    RE = ['0000011', '0000016', '00000147', '00000155', '00000158', '00000107', '00000239']
    LT = ['0000038', '0000044', '0000046', '0000047', '00000238', '00000177', '00000188', '00000181']
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = []
    MSE_LT = []
    MSE_LL = []
    MSE_SF = []
    MSE_LF = []

    torch.cuda.empty_cache()
    model.eval()
    k = 0
    flag = 0

    gif_path = './128_diff'
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    with torch.no_grad():
        # compute metrics over the dataset
        if manager.dataloaders["test"] is not None:
            # loss status and val status initial
            manager.reset_loss_status()
            manager.reset_metric_status("test")
            with tqdm(total=len(manager.dataloaders['test']), ncols=100) as t:
                for data_batch in manager.dataloaders["test"]:

                    video_name = data_batch["video_names"]
                    save_name = data_batch["save_name"]
                    imgs_full = data_batch["imgs_full"]

                    b, c, h, w = imgs_full.shape

                    data_batch = utils.tensor_gpu(data_batch)
                    output_batch = model(data_batch, flag)

                    # Homo_b = output_batch["Homo_b"]

                    flag += 1
                    t.update()
                    eval_results = compute_eval_results(data_batch, output_batch)
                    err_avg = eval_results
                    for j in range(len(err_avg)):

                        # img1, img2 = imgs_full[j, :3, :, :].permute(1, 2, 0).numpy().astype('uint8'), \
                        #              imgs_full[j, 3:, :, :].permute(1, 2, 0).numpy().astype('uint8')
                        # H_matrix = Homo_b[j, :, :].detach().cpu().numpy()

                        # img1_warp = cv2.warpPerspective(img1, H_matrix, (w, h))

                        # img1_warp = cv2.cvtColor(img1_warp, cv2.COLOR_BGR2RGB)
                        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        # imageio.mimsave(os.path.join(gif_path, '{}.gif'.format(save_name[j])), [img1_warp, img2], 'GIF', duration=0.5)

                        k += 1
                        if video_name[j] in RE:
                            MSE_RE.append(err_avg[j])
                        elif video_name[j] in LT:
                            MSE_LT.append(err_avg[j])
                        elif video_name[j] in LL:
                            MSE_LL.append(err_avg[j])
                        elif video_name[j] in SF:
                            MSE_SF.append(err_avg[j])
                        elif video_name[j] in LF:
                            MSE_LF.append(err_avg[j])

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg + MSE_LF_avg) / 5

            Metric = {
                "MSE_RE_avg": MSE_RE_avg,
                "MSE_LT_avg": MSE_LT_avg,
                "MSE_LL_avg": MSE_LL_avg,
                "MSE_SF_avg": MSE_SF_avg,
                "MSE_LF_avg": MSE_LF_avg,
                "AVG": MSE_avg
            }
            manager.update_metric_status(metrics=Metric, split="test")

            # update data to tensorboard
            manager.logger.info("Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} ".format(
                manager.epoch_val, MSE_avg, MSE_RE_avg, MSE_LT_avg, MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("test", title="test", color="red")


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # Only load model weights
    params.only_weights = True

    # Update args into params
    params.update(vars(args))

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

    # Define the model and optimizer
    if params.cuda:
        model = net.fetch_net(params).cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    else:
        model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[kwargs])
    model, optimizer, dataloaders["train"], dataloaders["val"], dataloaders["test"], scheduler = accelerator.prepare(
        model, optimizer, dataloaders["train"], dataloaders["val"], dataloaders["test"], scheduler)

    # initial status for checkpoint manager
    manager = Manager(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      params=params,
                      dataloaders=dataloaders,
                      writer=None,
                      logger=logger,
                      accelerator=accelerator)

    # Initial status for checkpoint manager

    # Reload weights from the saved file
    manager.load_checkpoints()

    # Test the model
    logger.info("Starting test")

    # Evaluate
    test(model, manager)
