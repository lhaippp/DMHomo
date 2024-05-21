import os
import cv2
import torch
import random
import pickle
import imageio
import logging

import numpy as np
import torch.utils.data

from HEM.utils_operations.flow_and_mapping_operations import convert_mapping_to_flow, from_homography_to_pixel_wise_mapping

from torch.utils.data import DataLoader, Dataset, ConcatDataset


def worker_init_fn(worker_id):
    rand_seed = random.randint(0, 2**32 - 1)
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def homo_scale(h0, w0, H, h1, w1):
    M_0 = np.array([[w0 / 2.0, 0., w0 / 2.0], [0., h0 / 2.0, h0 / 2.0],
                    [0., 0., 1.]])
    M_0_inv = np.linalg.inv(M_0)
    H_0_norm = np.matmul(np.matmul(M_0_inv, H), M_0)

    M_1 = np.array([[w1 / 2.0, 0., w1 / 2.0], [0., h1 / 2.0, h1 / 2.0],
                    [0., 0., 1.]])
    M_1_inv = np.linalg.inv(M_1)
    H_1 = np.matmul(np.matmul(M_1, H_0_norm), M_1_inv)
    return H_1


def homo_convert_to_flow(H, size=(360, 640)):

    mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
        size, H)

    mapping_from_homography_numpy = np.dstack(
        (mapping_from_homography_x, mapping_from_homography_y))
    flow = convert_mapping_to_flow(
        torch.from_numpy(mapping_from_homography_numpy).unsqueeze(0).permute(
            0, 3, 1, 2))
    return flow.detach().cpu().requires_grad_(False)


def make_align_heatmap(img1, img2):
    heat_map = img1.astype(np.float32) - img2.astype(np.float32)
    heat_map = abs(heat_map)
    heat_map = (heat_map - heat_map.min()) / heat_map.max()
    heat_map = np.where(heat_map < 0.1, 0, heat_map)
    heat_map = cv2.applyColorMap((heat_map * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET).astype(np.uint8)
    return heat_map

class HomoTestData(Dataset):

    def __init__(self, params, phase):
        assert phase in ["test", "val"]

        self.crop_size = params.crop_size
        self.patch_size_h, self.patch_size_w = params.crop_size
        self.generate_size = params.generate_size

        self.data_list = os.path.join(params.test_data_dir, "test.txt")
        self.npy_path = os.path.join(params.test_data_dir, "pt")
        self.image_path = os.path.join(params.test_data_dir, "img")
        self.data_infor = open(self.data_list, 'r').readlines()

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        print(f"HomoTestData length {len(self.data_infor)}")

    def __len__(self):
        return len(self.data_infor)

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')
        video_names = img_names.split('/')[0]
        img_names = img_names.split(' ')
        pt_names = img_names[0].split('/')[-1] + '_' + img_names[1].split(
            '/')[-1] + '.npy'
        save_name = img_names[0].split('.')[0].split(
            '/')[1] + '_' + img_names[1].split('.')[0].split('/')[1]

        img1 = cv2.imread(os.path.join(self.image_path, img_names[0]))
        img2 = cv2.imread(os.path.join(self.image_path, img_names[1]))

        img1_rgb, img2_rgb = img1, img2
        imgs_rgb_full = torch.cat(
            (torch.Tensor(img1_rgb), torch.Tensor(img2_rgb)), dim=-1).permute(
                2, 0, 1).float() / 255.

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                              dim=-1).permute(2, 0, 1).float()
        ori_h, ori_w, _ = img1.shape

        # 模拟diffusion生成数据的256尺寸
        # img1 = cv2.resize(img1, (self.generate_size, self.generate_size))
        # img2 = cv2.resize(img2, (self.generate_size, self.generate_size))
        # print(f'img1 shape {img1.shape}')

        pt_set = np.load(os.path.join(self.npy_path, pt_names),
                         allow_pickle=True).item()
        pt_set = torch.Tensor(pt_set["matche_pts"][:6])

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        # 模拟diffusion生成数据从 256 -> 原尺寸
        # img1 = cv2.resize(img1, (ori_w, ori_h)).unsqueeze(0)
        # img2 = cv2.resize(img2, (ori_w, ori_h)).unsqueeze(0)

        img1, img2, img1_rs, img2_rs = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0,
                                                                 1).float()
        imgs_gray_patch = torch.cat(
            (img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()

        ori_size = torch.Tensor([ori_w, ori_h]).float()
        Ph, Pw = img1_rs.size()

        pts = torch.Tensor([[0, 0], [Pw - 1, 0], [0, Ph - 1],
                            [Pw - 1, Ph - 1]]).float()
        start = torch.Tensor([0, 0]).reshape(2, 1, 1).float()

        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_rgb_full": imgs_rgb_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_gray_patch,
            "ori_size": ori_size,
            "pt_set": pt_set,
            'pt_names': pt_names,
            "video_names": video_names,
            "pts": pts,
            "start": start,
            "save_name": save_name,
            "ganhomo_mask": torch.ones_like(imgs_full),
        }
        return data_dict


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def fetch_dataloader(params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        status_manager: (class) status_manager

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    # for psedo labels generating
    # train_ds = GanHomoTrainData(params, phase='em2')
    # train_ds = UnHomoTrainData(params, phase='train')
    # train_ds = ConcatDataset([UnHomoTrainData(
    #     params, phase='train'), UnHomoTrainDataSup(params, phase='train')])
    val_ds = HomoTestData(params, phase='val')
    test_ds = HomoTestData(params, phase='test')

    dataloaders = {}
    # add defalt train data loader
    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=params.train_batch_size,
    #     shuffle=True,
    #     num_workers=params.num_workers,
    #     pin_memory=params.cuda,
    #     drop_last=True,
    #     worker_init_fn=worker_init_fn,
    #     collate_fn=collate_fn,
    # )
    dataloaders["train"] = None

    # chosse val or test data loader for evaluate
    for split in ["val", "test"]:
        if split in params.eval_type:
            if split == "val":
                dl = DataLoader(
                    val_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=params.cuda,
                )
            elif split == "test":
                dl = DataLoader(
                    test_ds,
                    batch_size=params.eval_batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=params.cuda,
                )
            else:
                raise ValueError(
                    "Unknown eval_type in params, should in [val, test]")
            dataloaders[split] = dl
        else:
            dataloaders[split] = None

    return dataloaders
