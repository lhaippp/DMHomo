import os
import torch

from termcolor import colored
from collections import defaultdict

from HEM.common import utils

class Manager():

    def __init__(self, model, optimizer, scheduler, params, dataloaders,
                 writer, logger, accelerator):
        # params status
        self.params = params

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.writer = writer
        self.logger = logger
        self.accelerator = accelerator

        self.epoch = 0
        self.step = 0
        self.best_val_score = float('inf')
        self.best_test_score = float('inf')
        self.cur_val_score = 0
        self.cur_test_score = 0
        self.epoch_val = 0

        # train status
        self.train_status = defaultdict(utils.AverageMeter)

        # val status
        self.val_status = defaultdict(utils.AverageMeter)

        # test status
        self.test_status = defaultdict(utils.AverageMeter)

        # model status
        self.loss_status = defaultdict(utils.AverageMeter)

    def update_step(self):
        self.step += 1

    def update_epoch(self):
        self.epoch += 1
        self.epoch_val += 1

    def update_loss_status(self, loss, split):
        if split == "train":
            for k, v in loss.items():
                self.loss_status[k].update(val=v.item(),
                                           num=self.params.train_batch_size)
        elif split == "val":
            for k, v in loss.items():
                self.loss_status[k].update(val=v.item(),
                                           num=self.params.eval_batch_size)
        elif split == "test":
            for k, v in loss.items():
                self.loss_status[k].update(val=v.item(),
                                           num=self.params.eval_batch_size)
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def update_metric_status(self, metrics, split):
        if split == "val":
            for k, v in metrics.items():
                self.val_status[k].update(val=v.item(),
                                          num=self.params.eval_batch_size)
                self.cur_val_score = self.val_status[
                    self.params.major_metric].avg
        elif split == "test":
            for k, v in metrics.items():
                self.test_status[k].update(val=v.item(),
                                           num=self.params.eval_batch_size)
                self.cur_test_score = self.test_status[
                    self.params.major_metric].avg
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def reset_loss_status(self):
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        if split == "val":
            for k, v in self.val_status.items():
                self.val_status[k].reset()
        elif split == "test":
            for k, v in self.test_status.items():
                self.test_status[k].reset()
        else:
            raise ValueError("Wrong eval type: {}".format(split))

    def print_train_info(self):
        exp_name = self.params.model_dir.split('/')[-1]
        print_str = "{} Epoch: {:2d}, lr={:.6f} ".format(
            exp_name, self.epoch,
            self.scheduler.get_last_lr()[0])
        print_str += " sup: %.4f(%.4f)" % (self.loss_status['supervise'].val,
                                           self.loss_status['supervise'].avg)
        print_str += " mask: %.4f(%.4f)" % (self.loss_status['mask_reg'].val,
                                            self.loss_status['mask_reg'].avg)
        print_str += " unsup: %.4f(%.4f)" % (self.loss_status['unsup'].val,
                                             self.loss_status['unsup'].avg)
        print_str += " nll: %.4f(%.4f)" % (self.loss_status['nll'].val,
                                           self.loss_status['nll'].avg)
        print_str += " total: %.4f(%.4f)" % (self.loss_status['total'].val,
                                             self.loss_status['total'].avg)
        return print_str

    def print_metrics(self, split, title="Eval", color="red"):
        if split == "val":
            metric_status = self.val_status
        elif split == "test":
            metric_status = self.test_status
        else:
            raise ValueError("Wrong eval type: {}".format(split))
        print_str = " | ".join("{}: {:.4f}".format(k, v.avg)
                               for k, v in metric_status.items())
        self.logger.info(
            colored("{} Results: {}".format(title, print_str),
                    color,
                    attrs=["bold"]))

    def check_best_save_last_checkpoints(self, latest_freq=5):
        state = {
            "state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
        }
        if "val" in self.dataloaders:
            state["best_val_score"] = self.best_val_score
        if "test" in self.dataloaders:
            state["best_test_score"] = self.best_test_score

        # save latest checkpoint
        if self.epoch % latest_freq == 0:
            latest_ckpt_name = os.path.join(self.params.model_dir,
                                            "model_latest.pth")
            if self.params.save_mode == "local":
                torch.save(state, latest_ckpt_name)
            else:
                raise NotImplementedError
            self.logger.info(
                "Saved latest checkpoint to: {}".format(latest_ckpt_name))

        # save val the latest metrics, and check if val is the best checkpoints
        if "val" in self.dataloaders:
            val_latest_metrics_name = os.path.join(self.params.model_dir,
                                                   "val_metrics_latest.json")
            utils.save_dict_to_json(self.val_status, val_latest_metrics_name)
            is_best = self.cur_val_score < self.best_val_score
            if is_best:
                # save metrics
                self.best_val_score = self.cur_val_score
                best_metrics_name = os.path.join(self.params.model_dir,
                                                 "val_metrics_best.json")
                utils.save_dict_to_json(self.val_status, best_metrics_name)
                self.logger.info("Current is val best, score={:.4f}".format(
                    self.best_val_score))
                # save checkpoint
                best_ckpt_name = os.path.join(
                    self.params.model_dir,
                    "val_model_best_{:.4f}.pth".format(self.best_val_score))
                if self.params.save_mode == "local":
                    torch.save(state, best_ckpt_name)
                else:
                    raise NotImplementedError
                self.logger.info(
                    "Saved val best checkpoint to: {}".format(best_ckpt_name))

        # save test latest metrics, and check if test is the best checkpoints
        # if self.dataloaders["test"] is not None:
        if "test" in self.dataloaders:
            test_latest_metrics_name = os.path.join(
                self.params.model_dir, "test_metrics_latest.json")
            utils.save_dict_to_json(self.test_status, test_latest_metrics_name)
            is_best = self.cur_test_score > self.best_test_score
            if is_best:
                # save metrics
                self.best_test_score = self.cur_test_score
                best_metrics_name = os.path.join(self.params.model_dir,
                                                 "test_metrics_best.json")
                utils.save_dict_to_json(self.test_status, best_metrics_name)
                self.logger.info("Current is test best, score={:.4f}".format(
                    self.best_test_score))
                # save checkpoint
                best_ckpt_name = os.path.join(self.params.model_dir,
                                              "test_model_best.pth")
                if self.params.save_mode == "local":
                    torch.save(state, best_ckpt_name)
                else:
                    raise NotImplementedError
                self.logger.info(
                    "Saved test best checkpoint to: {}".format(best_ckpt_name))

    def load_checkpoints(self):
        if self.params.cuda:
            state = torch.load(self.params.restore_file,
                               map_location=self.accelerator.device)
        else:
            state = torch.load(self.params.restore_file,
                               map_location=torch.device('cpu'))

        ckpt_component = []
        if "state_dict" in state and self.model is not None:
            try:
                model = self.accelerator.unwrap_model(self.model)
                model.load_state_dict(state["state_dict"], True)
                # self.model.load_state_dict(state["state_dict"])
            except:
                # raise Exception('cannot load model!')
                self.accelerator.print("Using custom loading state_dict")
                model = self.accelerator.unwrap_model(self.model)
                net_dict = model.state_dict()
                if "module" not in list(state["state_dict"].keys())[0]:
                    state_dict = {
                        "module." + k: v
                        for k, v in state["state_dict"].items()
                        if "module." + k in net_dict.keys()
                    }
                else:
                    state_dict = {
                        k.replace("module.", ""): v
                        for k, v in state["state_dict"].items()
                        if k.replace("module.", "") in net_dict.keys()
                    }
                net_dict.update(state_dict)
                model.load_state_dict(net_dict)

            ckpt_component.append("net")

        if "HNet_state_dict" in state:
            try:
                self.model.load_state_dict(state["HNet_state_dict"], False)
            except:
                self.accelerator.print("Using custom loading HNet")
                net_dict = self.model.state_dict()
                if "module" not in list(state["HNet_state_dict"].keys())[0]:
                    state_dict = {
                        "module." + k: v
                        for k, v in state["HNet_state_dict"].items()
                        if "module." + k in net_dict.keys()
                    }
                else:
                    state_dict = {
                        k.replace("module.", ""): v
                        for k, v in state["HNet_state_dict"].items()
                        if k.replace("module.", "") in net_dict.keys()
                    }
                net_dict.update(state_dict)
                self.model.load_state_dict(net_dict)

            ckpt_component.append("HNet")

        if not self.params.only_weights:
            if "optimizer" in state and self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(state["optimizer"], )

                except Warning("Using custom loading optimizer"):
                    raise Exception('cannot load optimizer!')
                    # optimizer_dict = self.optimizer.state_dict()
                    # state_dict = {k: v for k, v in state["optimizer"].items() if k in optimizer_dict.keys()}
                    # optimizer_dict.update(state_dict)
                    # self.optimizer.load_state_dict(optimizer_dict)

                ckpt_component.append("opt")

            if "scheduler" in state and self.train_status[
                    "scheduler"] is not None:
                try:
                    self.scheduler.load_state_dict(state["scheduler"])

                except Warning("Using custom loading scheduler"):
                    raise Exception('cannot load scheduler!')
                    # scheduler_dict = self.scheduler.state_dict()
                    # state_dict = {k: v for k, v in state["scheduler"].items() if k in scheduler_dict.keys()}
                    # scheduler_dict.update(state_dict)
                    # self.scheduler.load_state_dict(scheduler_dict)

                ckpt_component.append("sch")

            if "step" in state:
                self.train_status["step"] = state["step"] + 1
                self.step = state["step"]
                ckpt_component.append("step")

            if "epoch" in state:
                self.train_status["epoch"] = state["epoch"] + 1
                self.epoch = state["epoch"]
                ckpt_component.append("epoch")

            if "best_val_score" in state:
                self.best_val_score = state["best_val_score"]
                ckpt_component.append("best val score: {:.3g}".format(
                    self.best_val_score))

            if "best_test_score" in state:
                self.best_test_score = state["best_test_score"]
                ckpt_component.append("best test score: {:.3g}".format(
                    self.best_test_score))

        ckpt_component = ", ".join(i for i in ckpt_component)
        self.logger.info("Loaded models from: {}".format(
            self.params.restore_file))
        self.logger.info("Ckpt load: {}".format(ckpt_component))
