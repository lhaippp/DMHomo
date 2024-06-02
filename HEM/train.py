"""Train the model"""

import argparse
import datetime
import os
import logging
import itertools
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1 ,2'
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import dataset.data_loader as data_loader
import model.net as net
from common import utils
from common.manager import Manager
from evaluate import evaluate
from loss.losses import compute_losses

import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/root/test/HEM/experiments/Bases_Ours/',
                    help="Directory containing params.json")
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--restore_file',
                    default='/data/supervise_homo/unsupervise_codes/'
                    'BasesHomo/experiments/base_model/best_0.5012.pth.tar',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'experiments/version1/model_latest.pth'
parser.add_argument('-ow', '--only_weights', action='store_true',
                    default=False, help='Only use weights to load or load all train status.')
parser.add_argument('--seed', type=int, default=230, help='random seed')


def cycle(dl):
    while True:
        for data in dl:
            yield data


def train(model, manager):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # loss status and val/test status initial
    manager.reset_loss_status()
    # set model to training mode
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    model.train()

    while manager.step < manager.total_step:
        data_batch = next(manager.dataloaders['train'])
        manager.optimizer.zero_grad()

        # infor print
        print_str = manager.print_train_info()

        # compute model output and loss
        output = model(data_batch, manager.step)

        loss = {}
        loss.update(compute_losses(data_batch, output, manager.params))

        manager.update_loss_status(loss=loss, split="train")

        manager.accelerator.backward(loss['total'])
        manager.optimizer.step()

        # save loss in trainlog
        manager.update_step()
        if manager.step % manager.train_data_length == 0 and manager.step != 0:
            manager.scheduler.step()
            manager.update_epoch()

        manager.t.set_description(desc=print_str)
        manager.t.update()

        if manager.step % manager.train_num_steps == 0 and manager.step != 0:
            break


def train_and_evaluate(model, manager):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    """

    # reload weights from restore_file if specified
    if args.resume:
        # the checkpoint path is set by args.restore_file
        manager.load_checkpoints()

    manager.train_num_steps = manager.params.train_num_steps
    manager.total_step = manager.params.num_epochs * manager.train_data_length

    # for epoch in range(manager.params.num_epochs):
    with tqdm(
            total=manager.total_step,
            ncols=200,
            disable=not manager.accelerator.is_main_process,
            initial=manager.step,
    ) as t:
        manager.t = t
        while manager.step < manager.total_step:
            # compute number of batches in one epoch (one full pass over the training set)
            train(model, manager)

            # if manager.accelerator.is_main_process:
            evaluate(model, manager)

            manager.accelerator.wait_for_everyone()
            if manager.accelerator.is_main_process:
                manager.check_best_save_last_checkpoints(latest_freq=1)

    manager.accelerator.print('training complete')


if __name__ == '__main__':
    # Logging Configuration
    # logging.getLogger('boto').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)

    # Load the parameters from json file
    # torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Update args into params
    params.update(vars(args))

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    # torch.manual_seed(args.seed)
    # if params.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # Set the logger
    logger = utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Loading the datasets from {}".format(params.train_data_dir))

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(params)

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

    manager.train_data_length = len(dataloaders["train"])
    dataloaders["train"] = cycle(dataloaders["train"])
    # Train the model
    logger.info("Starting training for {} epoch(s)".format(params.num_epochs))

    train_and_evaluate(model, manager)
