import os
import cv2
import json
import imageio
import logging
import shutil

import torch
import coloredlogs


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, dict):
        """Loads parameters from json file"""
        self.__dict__.update(dict)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class AverageMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_previous = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val_previous = self.val
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def loss_meter_manager_intial(loss_meter_names):
    # 用于根据meter名字初始化需要用到的loss_meter
    loss_meters = []
    for name in loss_meter_names:
        exec("%s = %s" % (name, 'AverageMeter()'))
        exec("loss_meters.append(%s)" % name)

    return loss_meters


def tensor_gpu(batch, check_on=True):

    def check_on_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            tensor_g = tensor_
        else:
            tensor_g = tensor_.cuda()
        return tensor_g

    def check_off_gpu(tensor_):
        if isinstance(tensor_, str) or isinstance(tensor_, list):
            return tensor_

        if tensor_.is_cuda:
            tensor_c = tensor_.cpu()
        else:
            tensor_c = tensor_
        tensor_c = tensor_c.detach().numpy()
        return tensor_c

    if torch.cuda.is_available():
        if check_on:
            for k, v in batch.items():
                batch[k] = check_on_gpu(v)
        else:
            for k, v in batch.items():
                batch[k] = check_off_gpu(v)
    else:
        if check_on:
            batch = batch
        else:
            for k, v in batch.items():
                batch[k] = v.detach().numpy()

    return batch


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # if not logger.handlers:
    #     # Logging to a file
    #     file_handler = logging.FileHandler(log_path)
    #     file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    #     logger.addHandler(file_handler)
    #
    #     # Logging to console
    #     stream_handler = logging.StreamHandler()
    #     stream_handler.setFormatter(logging.Formatter('%(message)s'))
    #     logger.addHandler(stream_handler)

    coloredlogs.install(level='INFO',
                        logger=logger,
                        fmt='%(asctime)s %(name)s %(message)s')
    file_handler = logging.FileHandler(log_path)
    log_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info('Output and logs will be saved to {}'.format(log_path))
    return logger


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    save_dict = {}
    with open(json_path, "w") as f:
        # We need to convert the values to float for json (it doesn"t accept np.array, np.float, )
        for k, v in d.items():
            if isinstance(v, AverageMeter):
                save_dict[k] = float(v.avg)
            else:
                save_dict[k] = float(v)
        json.dump(save_dict, f, indent=4)


def make_gif(img1, img2, exp_name, name):
    if not os.path.exists(f"gif_results/{exp_name}"):
        os.mkdir(f"gif_results/{exp_name}")
    img1, img2 = cv2.imread(img1), cv2.imread(img2)
    with imageio.get_writer(f'gif_results/{exp_name}/{name}.gif',
                            mode='I',
                            duration=0.5) as writer:
        writer.append_data(img1)
        writer.append_data(img2)
