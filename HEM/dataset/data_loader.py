import os
import cv2
import torch
import random
import pickle
import imageio
import logging

import numpy as np
import torch.utils.data
import nori2 as nori

from HEM.utils_operations.flow_and_mapping_operations import convert_mapping_to_flow, from_homography_to_pixel_wise_mapping

from torch.utils.data import DataLoader, Dataset, ConcatDataset

fetcher = nori.Fetcher()


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


class UnHomoTrainData(Dataset):

    def __init__(self, params, phase='train'):
        assert phase in ['train', 'val', 'test']
        # 参数预设

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.params = params
        self.crop_size = self.params.crop_size
        self.ori_h, self.ori_w = self.params.ori_size[0], self.params.ori_size[
            1]
        self.rho = self.params.rho

        self.data_dir = self.params.train_data_dir
        self.nori_pkl = pickle.load(
            open(
                os.path.join(self.data_dir,
                             f'{self.params.nori_dataset_name}.pickle'), 'rb'))
        self.nori_pkl_value_list = list(self.nori_pkl.values())
        # print(self.nori_pkl)
        # self.imgs_dtype = self.nori_pkl_value_list[0]["_id_1_dtype"]
        # self.imgs_shape = self.nori_pkl_value_list[0]["_id_1_shape"]
        # self.homos_dtype = self.nori_pkl_value_list[0]["_id_2_dtype"]
        # self.homos_shape = self.nori_pkl_value_list[0]["_id_2_shape"]
        self.imgs_shape = self.nori_pkl["img12_shape"]
        # print(f"self.homos_dtype {self.homos_dtype}")
        # self.images_dir = os.path.join(self.data_dir, 'images')
        # self.homo_dir = os.path.join(self.data_dir, 'homo')

        self.seed = 0
        random.seed(self.seed)

        print(
            f"UnHomoTrainData load {len(self.nori_pkl_value_list)} data from oss!"
        )
        self.cnt = 0

    def __len__(self):
        return len(self.nori_pkl_value_list)

    def bytes2np(self, data, dtype, c=3, h=64, w=64):
        data = np.fromstring(data, dtype)
        data = data.reshape((c, h, w)) if c else data.reshape((h, w))
        return data

    def __getitem__(self, idx):
        # img loading
        nori_pkl = self.nori_pkl_value_list[idx]
        try:
            data = fetcher.get(nori_pkl["img12_homo12_nid"])
        except:
            return None

        data = pickle.loads(data)
        images, homo_gt = data["img"], data["homo"]
        images = images.transpose(1, 2, 0)
        img1, img2 = images[:, :, :3], images[:, :, 3:6]

        imgs_rgb_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                                  dim=-1).permute(2, 0, 1).float() / 255.

        # unit test for async nori
        # self.cnt += 1
        # img1_warp = cv2.warpPerspective(img1, homo_gt, (256, 256))
        # imageio.mimsave(
        #     f"unit_test/test_async_nori_{self.cnt}.gif",
        #     [
        #         np.concatenate((img1, img1_warp), 1)[:, :, ::-1],
        #         np.concatenate((img2, img2), 1)[:, :, ::-1],
        #     ],
        #     'GIF',
        #     duration=0.5,
        # )
        # if self.cnt > 10:
        #     raise Exception

        img1_warp = cv2.warpPerspective(img1, homo_gt, (256, 256))

        identity_heatmap = make_align_heatmap(img1, img2)
        warping_heatmap = make_align_heatmap(img1_warp, img2)
        cv2.imwrite(
            f'generate_results/buf_{idx}.png',
            np.concatenate((img1, img2, identity_heatmap, warping_heatmap), 1),
        )

        h, w, c = img1.shape

        homo_gt = homo_scale(h, w, homo_gt, self.ori_h, self.ori_w)
        homo_gt_inv = np.linalg.inv(homo_gt)

        img1, img2 = cv2.resize(img1, (self.ori_w, self.ori_h)), cv2.resize(
            img2, (self.ori_w, self.ori_h))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start = self.data_aug(
            img1, img2, homo_gt, homo_gt_inv)

        imgs_gray_full = torch.cat((img1, img2), dim=2).permute(2, 0,
                                                                1).float()
        imgs_gray_patch = torch.cat((img1_patch, img2_patch),
                                    dim=2).permute(2, 0, 1).float()
        flow_gt_full = torch.cat((flow_gt_b, flow_gt_f), dim=1).squeeze(0)
        flow_gt_patch = torch.cat((flow_gt_b_patch, flow_gt_f_patch),
                                  dim=1).squeeze(0)
        start = torch.Tensor(start).reshape(2, 1, 1).float()
        # output dict
        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_gray_patch": imgs_gray_patch,
            "flow_gt_full": flow_gt_full,
            "flow_gt_patch": flow_gt_patch,
            "start": start,
            "ganhomo_mask": torch.ones_like(imgs_gray_patch),
            "imgs_rgb_full": imgs_rgb_full,
        }

        return data_dict

    def data_aug(self,
                 img1,
                 img2,
                 homo_gt,
                 homo_gt_inv,
                 start=None,
                 normalize=True,
                 gray=True):

        def random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start):
            height, width = img1.shape[:2]
            patch_size_h, patch_size_w = self.crop_size

            if start is None:
                x = random.randint(self.rho, width - self.rho - patch_size_w)
                y = random.randint(self.rho, height - self.rho - patch_size_h)
                start = [x, y]
            else:
                x, y = start
            img1_patch = img1[y:y + patch_size_h, x:x + patch_size_w, :]
            img2_patch = img2[y:y + patch_size_h, x:x + patch_size_w, :]

            flow_gt_b = homo_convert_to_flow(homo_gt_inv, (height, width))
            flow_gt_f = homo_convert_to_flow(homo_gt, (height, width))
            flow_gt_b_patch = flow_gt_b[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            flow_gt_f_patch = flow_gt_f[:, :, y:y + patch_size_h,
                                        x:x + patch_size_w]
            return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start

        if normalize:
            img1 = (img1 - self.mean_I) / self.std_I
            img2 = (img2 - self.mean_I) / self.std_I

        if gray:
            img1 = np.mean(img1, axis=2, keepdims=True)
            img2 = np.mean(img2, axis=2, keepdims=True)

        img1, img2 = list(map(torch.Tensor, [img1, img2]))

        img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start \
            = random_crop_tt(img1, img2, homo_gt, homo_gt_inv, start)

        return img1, img2, img1_patch, img2_patch, flow_gt_b, flow_gt_f, flow_gt_b_patch, flow_gt_f_patch, start


class UnHomoTrainDataSup(UnHomoTrainData):

    def __init__(self, params, phase='train'):
        super().__init__(params, phase)

        self.nori_pkl = pickle.load(
            open(
                os.path.join(self.data_dir,
                             f'{self.params.nori_dataset_name_sup}.pickle'),
                'rb'))
        self.nori_pkl_value_list = list(self.nori_pkl.values())

        print(
            f"UnHomoTrainDataSup load {len(self.nori_pkl_value_list)} data from oss!"
        )


class GanHomoTrainData(Dataset):

    def __init__(
        self,
        params,
        phase,
        total_data_slice_idx=1,
        data_slice_idx=0,
    ):
        assert phase in ['em2']
        # 参数预设
        self.homo = np.load(
            '/data/denoising-diffusion-pytorch/work/20230227.9i6o.GanHomo.GanMask.Dilate.Class.AdpL/dataset/HomoGAN_sup.npy',
            allow_pickle=True).item()

        self.nori_info = np.load(
            '/data/denoising-diffusion-pytorch/work/20221230.cond.rgbHomoFlow.9i6o/dataset/train.npy',
            allow_pickle=True).item()
        self.nori_ids, self.nori_length, self.plk_dict = self.nori_info[
            "nori_ids"], self.nori_info["nori_length"], self.nori_info[
                "plk_dict"]

        self.slice_nori_length = (self.nori_length // total_data_slice_idx)
        print(f"self.nori_length {self.slice_nori_length}/{self.nori_length}")
        self.nori_ids = self.nori_ids[data_slice_idx *
                                      self.slice_nori_length:(data_slice_idx +
                                                              1) *
                                      self.slice_nori_length]

        from torchvision import transforms as T
        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
        ])

        self.mean_I = np.array([118.93, 113.97, 102.60]).reshape(1, 1, 3)
        self.std_I = np.array([69.85, 68.81, 72.45]).reshape(1, 1, 3)

        self.crop_size = params.crop_size

    @staticmethod
    def bytes2np(data, size, dtype=np.float32):
        data = np.fromstring(data, dtype)
        data = data.reshape(size)
        return data

    def __len__(self):
        print(f"The length of UnHomoTrainData is {self.slice_nori_length}")
        return self.slice_nori_length

    def __getitem__(self, idx):
        # img loading
        images_bytes = fetcher.get(self.nori_ids[idx])
        buf = self.plk_dict[self.nori_ids[idx]]

        images_dtype, images_size = buf['_images_dtype'], buf['_images_size']
        masks_id, masks_dtype, masks_size, file_name = \
            buf['_id_masks'], buf['_masks_dtype'], buf['_masks_size'], buf['idx']
        masks_bytes = fetcher.get(masks_id)

        images = self.bytes2np(images_bytes,
                               size=images_size,
                               dtype=images_dtype).astype(np.float32)
        masks = self.bytes2np(masks_bytes, size=masks_size, dtype=masks_dtype)

        # uint8 [0, 256]
        img1, img2 = images[:, :, :3], images[:, :, 3:]

        # homoGAN mask
        mask = masks[:, :, 1:2].astype(np.float32)
        mask = torch.Tensor(mask)

        img1_rgb, img2_rgb = img1, img2

        imgs_rgb_full = torch.cat(
            (torch.Tensor(img1_rgb), torch.Tensor(img2_rgb)), dim=-1).permute(
                2, 0, 1).float() / 255.

        imgs_full = torch.cat((torch.Tensor(img1), torch.Tensor(img2)),
                              dim=-1).permute(2, 0, 1).float()

        img1 = (img1 - self.mean_I) / self.std_I
        img2 = (img2 - self.mean_I) / self.std_I

        img1 = np.mean(img1, axis=2, keepdims=True)
        img2 = np.mean(img2, axis=2, keepdims=True)

        img1_rs = cv2.resize(img1, (self.crop_size[1], self.crop_size[0]))
        img2_rs = cv2.resize(img2, (self.crop_size[1], self.crop_size[0]))

        img1, img2, img1_rs, img2_rs = list(
            map(torch.Tensor, [img1, img2, img1_rs, img2_rs]))

        imgs_gray_full = torch.cat((img1, img2), dim=-1).permute(2, 0,
                                                                 1).float()
        imgs_gray_patch = torch.cat(
            (img1_rs.unsqueeze(0), img2_rs.unsqueeze(0)), dim=0).float()

        data_dict = {
            "imgs_gray_full": imgs_gray_full,
            "imgs_rgb_full": imgs_rgb_full,
            "imgs_full": imgs_full,
            "imgs_gray_patch": imgs_gray_patch,
            "ganhomo_mask": mask,
        }
        return data_dict


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
