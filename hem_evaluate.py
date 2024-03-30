"""Evaluates the model"""

import os
import cv2
import torch
import imageio
import logging
import argparse
import torchvision

import numpy as np
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import HEM.model.net as net
import HEM.dataset.data_loader as data_loader

from HEM.common import utils
from HEM.common.manager import Manager
from HEM.model.utils import get_warp_flow
from HEM.loss.losses import compute_eval_results

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',
                    default='experiments/Bases_Ours/',
                    help="Directory containing params.json")
parser.add_argument(
    '--restore_file',
    default='experiments/Bases_Ours/val_model_best_0.6689.pth',
    help=
    "Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'experiments/version1/model_latest.pth'
parser.add_argument('-ow',
                    '--only_weights',
                    action='store_true',
                    default=True,
                    help='Only use weights to load or load all train status.')


def make_align_heatmap(ref, inp):
    heat_map = inp.astype(np.float32) - ref.astype(np.float32)
    heat_map = abs(heat_map)
    heat_map -= heat_map.min()
    heat_map /= heat_map.max()
    heat_map = np.clip(heat_map, 0.01, 1)
    heat_map = cv2.applyColorMap((heat_map * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)[:, :, ::-1].astype(np.uint8)
    # heat_map = append_text_information(heat_map, text, ssim_scores)
    return heat_map


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

    RE = [
        '0000011', '0000016', '00000147', '00000155', '00000158', '00000107',
        '00000239', '0000030'
    ]
    LT = [
        '0000038', '0000044', '0000046', '0000047', '00000238', '00000177',
        '00000188', '00000181'
    ]
    LL = ['0000085', '00000100', '0000091', '0000092', '00000216', '00000226']
    SF = ['00000244', '00000251', '0000026', '0000030', '0000034', '00000115']
    LF = ['00000104', '0000031', '0000035', '00000129', '00000141', '00000200']
    MSE_RE = [0]
    MSE_LT = [0]
    MSE_LL = [0]
    MSE_SF = [0]
    MSE_LF = [0]

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
                        # print(f"RE {err_avg[j]:.2f}/{sum(MSE_RE) / len(MSE_RE):.2f}")
                    elif video_name[j] in LT:
                        MSE_LT.append(err_avg[j])
                        # print(f"LT {err_avg[j]:.2f}/{sum(MSE_LT) / len(MSE_LT):.2f}")
                    elif video_name[j] in LL:
                        MSE_LL.append(err_avg[j])
                        # print(f"LL {err_avg[j]:.2f}/{sum(MSE_LL) / len(MSE_LL):.2f}")
                    elif video_name[j] in SF:
                        MSE_SF.append(err_avg[j])
                        # print(f"SF {err_avg[j]:.2f}/{sum(MSE_SF) / len(MSE_SF):.2f}")
                    elif video_name[j] in LF:
                        MSE_LF.append(err_avg[j])
                        # print(f"LF {err_avg[j]:.2f}/{sum(MSE_LF) / len(MSE_LF):.2f}")
                    print(f"RE {MSE_RE[-1]:.2f}/{sum(MSE_RE) / len(MSE_RE):.2f} | " \
                          f"LT {MSE_LT[-1]:.2f}/{sum(MSE_LT) / len(MSE_LT):.2f} | " \
                          f"LL {MSE_LL[-1]:.2f}/{sum(MSE_LL) / len(MSE_LL):.2f} | " \
                          f"SF {MSE_SF[-1]:.2f}/{sum(MSE_SF) / len(MSE_SF):.2f} | " \
                          f"LF {MSE_LF[-1]:.2f}/{sum(MSE_LF) / len(MSE_LF):.2f} |")

            MSE_RE_avg = sum(MSE_RE) / len(MSE_RE)
            MSE_LT_avg = sum(MSE_LT) / len(MSE_LT)
            MSE_LL_avg = sum(MSE_LL) / len(MSE_LL)
            MSE_SF_avg = sum(MSE_SF) / len(MSE_SF)
            MSE_LF_avg = sum(MSE_LF) / len(MSE_LF)

            # results shoud be gatherd from multiple GPUs to single GPU
            MSE_RE_avg = torch.mean(
                manager.accelerator.gather(
                    torch.tensor(MSE_RE_avg).cuda(
                        manager.accelerator.process_index)))
            MSE_LT_avg = torch.mean(
                manager.accelerator.gather(
                    torch.tensor(MSE_LT_avg).cuda(
                        manager.accelerator.process_index)))
            MSE_LL_avg = torch.mean(
                manager.accelerator.gather(
                    torch.tensor(MSE_LL_avg).cuda(
                        manager.accelerator.process_index)))
            MSE_SF_avg = torch.mean(
                manager.accelerator.gather(
                    torch.tensor(MSE_SF_avg).cuda(
                        manager.accelerator.process_index)))
            MSE_LF_avg = torch.mean(
                manager.accelerator.gather(
                    torch.tensor(MSE_LF_avg).cuda(
                        manager.accelerator.process_index)))

        if manager.accelerator.is_main_process:
            MSE_avg = (MSE_RE_avg + MSE_LT_avg + MSE_LL_avg + MSE_SF_avg +
                       MSE_LF_avg) / 5
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

            manager.logger.info(
                "Loss/valid epoch_val {}: {:.4f}. RE:{:.4f} LT:{:.4f} LL:{:.4f} SF:{:.4f} LF:{:.4f} "
                .format(manager.epoch_val, MSE_avg, MSE_RE_avg, MSE_LT_avg,
                        MSE_LL_avg, MSE_SF_avg, MSE_LF_avg))

            manager.print_metrics("val", title="val", color="green")

            manager.epoch_val += 1

            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)
            model.train()
            val_metrics = {'MSE_avg': MSE_avg}
            return val_metrics


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
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
    model = net.fetch_net(params)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.gamma)

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=True, kwargs_handlers=[kwargs])
    model, optimizer, dataloaders["val"], dataloaders[
        "test"], scheduler = accelerator.prepare(model, optimizer,
                                                 dataloaders["val"],
                                                 dataloaders["test"],
                                                 scheduler)

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
    # test(model, manager)
    evaluate(model, manager)
