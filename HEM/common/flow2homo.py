import torch
import numpy as np
import cv2
import os
import torch
import random

import kornia.geometry as ge
import torch.nn.functional as F


def get_grid(batch_size, H, W, start=0):

    # if torch.cuda.is_available():
    #     xx = torch.arange(0, W).cuda()
    #     yy = torch.arange(0, H).cuda()
    # else:
    #     xx = torch.arange(0, W)
    #     yy = torch.arange(0, H)
    xx = torch.arange(0, W)
    yy = torch.arange(0, H)

    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + \
        start  # add the coordinate of left top
    return grid


def homo_flow_gen(flow):
    b, c, h, w = flow.shape

    grid = get_grid(b, h, w)

    if flow.is_cuda:
        grid = grid.cuda()

    grid = grid.permute(0, 2, 3, 1).reshape(b, 1, -1, 2)

    flow = flow.permute(0, 2, 3, 1).reshape(b, 1, -1, 2)

    homo = ge.homography.find_homography_dlt(grid, grid + flow)
    # print(f'homo')
    # homo = DLT_solve(grid, flow)

    homo_flow = homo_convert_to_flow(homo, size=(h, w))

    return homo_flow


def flow2homo(flow, accelerator):
    flow = flow.permute(0, 3, 1, 2)
    b, c, h, w = flow.shape
    assert c == 2, f"shape of flow is not correct, as: [{flow.shape}]"

    grid = get_grid(b, h, w).to(flow.device)

    grid = grid.permute(0, 2, 3, 1).reshape(b, -1, 2)
    flow = flow.permute(0, 2, 3, 1).reshape(b, -1, 2)
    # print(f"grid shape - flow shape: {grid.shape} - {flow.shape}")
    # homo = DLT_solve(grid, flow)
    homo = ge.homography.find_homography_dlt(grid, grid + flow)

    # homo = []
    # for i in range(grid.shape[0]):
    #     grid_np = grid[i].detach().cpu().numpy()
    #     flow_np = flow[i].detach().cpu().numpy()
    #     print(f"grid_np:[{grid_np.shape}] - flow_np:[{flow_np.shape}]")
    #     homo.append(cv2.findHomography(grid_np.squeeze(), (grid_np + flow_np).squeeze())[0])
    # homo = np.concatenate(homo, 0)[None]
    # homo = torch.from_numpy(homo).to(accelerator.device)
    return homo


def DLT_solve(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)

    bs, _ = src_p.shape[:2]
    divide = int(np.sqrt(len(src_p[0]) / 2) - 1)
    row_num = (divide + 1) * 2
    src_ps = src_p
    off_sets = off_set
    for i in range(divide):
        for j in range(divide):
            h4p = src_p[:, [
                2 * j + row_num * i, 2 * j + row_num * i + 1, 2 * (j + 1) +
                row_num * i, 2 * (j + 1) + row_num * i + 1, 2 * (j + 1) +
                row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num +
                1, 2 * j + row_num * i + row_num, 2 * j + row_num * i +
                row_num + 1
            ]].reshape(bs, 1, 4, 2)

            pred_h4p = off_set[:, [
                2 * j + row_num * i, 2 * j + row_num * i + 1, 2 * (j + 1) +
                row_num * i, 2 * (j + 1) + row_num * i + 1, 2 * (j + 1) +
                row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num +
                1, 2 * j + row_num * i + row_num, 2 * j + row_num * i +
                row_num + 1
            ]].reshape(bs, 1, 4, 2)

            if i + j == 0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis=1)
                off_sets = torch.cat((off_sets, pred_h4p), axis=1)

    bs, n, h, w = src_ps.shape

    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, h, 1)
    if off_set.is_cuda:
        ones = ones.to(off_set.device)
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1)
    if off_set.is_cuda:
        zeros = zeros.to(off_set.device)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.linalg.pinv(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)

    return H


def convert_mapping_to_flow(mapping, output_channel_first=True):
    if not isinstance(mapping, np.ndarray):
        # torch tensor
        if len(mapping.shape) == 4:
            if mapping.shape[1] != 2:
                # load_size is BxHxWx2
                mapping = mapping.permute(0, 3, 1, 2)

            B, C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            # if mapping.is_cuda:
            #     grid = grid.cuda()
            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(0, 2, 3, 1)
        else:
            if mapping.shape[0] != 2:
                # load_size is HxWx2
                mapping = mapping.permute(2, 0, 1)

            C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            # attention, concat axis=0 here
            grid = torch.cat((xx, yy), 0).float()

            # if mapping.is_cuda:
            #     grid = grid.cuda()

            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1, 2, 0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(mapping.shape) == 4:
            if mapping.shape[3] != 2:
                # load_size is Bx2xHxW
                mapping = mapping.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = mapping.shape[:3]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if mapping.shape[0] == 2:
                # load_size is 2xHxW
                mapping = mapping.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = mapping.shape[:2]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def from_homography_to_pixel_wise_mapping(shape, H):
    """
    From a homography relating image I to image I', computes pixel wise mapping and pixel wise displacement
    between pixels of image I to image I'
    Args:
        shape: shape of image
        H: homography

    Returns:
        map_x mapping of each pixel of image I in the horizontal direction (given index of its future position)
        map_y mapping of each pixel of image I in the vertical direction (given index of its future position)
    """
    h_scale, w_scale = shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    X, Y = X.flatten(), Y.flatten()
    # X is same shape as shape, with each time the horizontal index of the pixel

    # create matrix representation --> each contain horizontal coordinate, vertical and 1
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

    # multiply Hinv to XYhom to find the warped grid
    XYwarpHom = np.dot(H, XYhom)
    Xwarp = XYwarpHom[0, :] / (XYwarpHom[2, :] + 1e-8)
    Ywarp = XYwarpHom[1, :] / (XYwarpHom[2, :] + 1e-8)

    # reshape to obtain the ground truth mapping
    map_x = Xwarp.reshape((h_scale, w_scale))
    map_y = Ywarp.reshape((h_scale, w_scale))
    return map_x.astype(np.float32), map_y.astype(np.float32)


def homo_convert_to_flow(H, size=(360, 640)):

    b = H.shape[0]
    synthetic_flow = []

    for b_ in range(b):
        mapping_from_homography_x, mapping_from_homography_y = from_homography_to_pixel_wise_mapping(
            size, H[b_].squeeze(0).detach().cpu().numpy())
        mapping_from_homography_numpy = np.dstack(
            (mapping_from_homography_x, mapping_from_homography_y))
        flow_gt_ = convert_mapping_to_flow(
            torch.from_numpy(mapping_from_homography_numpy).unsqueeze(
                0).permute(0, 3, 1, 2))

        synthetic_flow.append(flow_gt_)

    flow = torch.cat(synthetic_flow, dim=0)

    return flow.to(H.device).requires_grad_(True)


def upsample2d_flow_as(inputs,
                       target_as,
                       mode="bilinear",
                       if_rate=False,
                       align_corners=True):
    _, _, h, w = target_as.size()
    if if_rate:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
    if mode == 'nearest':
        res = F.interpolate(inputs, [h, w], mode=mode)
    else:
        res = F.interpolate(inputs, [h, w],
                            mode=mode,
                            align_corners=align_corners)
    return res
