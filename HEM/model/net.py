import os
import math
import torch
import warnings

import numpy as np
import torch.nn as nn
import kornia.geometry as ge
import torch.nn.functional as F

from .swin_multi import SwinTransformer
from timm.models.layers import trunc_normal_

from .utils import get_warp_flow, upsample2d_flow_as, get_grid, make_gif
from einops import rearrange
from torchvision import utils

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ['Net']


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

    Ainv = torch.linalg.inv(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)

    return H

def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilation(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    dilation, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilation


def gen_basis(h, w, is_qr=True, is_scale=True):
    basis_nb = 8
    grid = get_grid(1, h, w).permute(0, 2, 3,
                                     1).contiguous()  # 1, w, h, (x, y, 1)
    flow = grid[:, :, :, :2] * 0

    names = globals()
    for i in range(1, basis_nb + 1):
        names['basis_' + str(i)] = flow.clone()

    basis_1[:, :, :, 0] += grid[:, :, :, 0]  # [1, w, h, (x, 0)]
    basis_2[:, :, :, 0] += grid[:, :, :, 1]  # [1, w, h, (y, 0)]
    basis_3[:, :, :, 0] += 1  # [1, w, h, (1, 0)]
    basis_4[:, :, :, 1] += grid[:, :, :, 0]  # [1, w, h, (0, x)]
    basis_5[:, :, :, 1] += grid[:, :, :, 1]  # [1, w, h, (0, y)]
    basis_6[:, :, :, 1] += 1  # [1, w, h, (0, 1)]
    basis_7[:, :, :, 0] += grid[:, :, :, 0]**2  # [1, w, h, (x^2, xy)]
    basis_7[:, :, :, 1] += grid[:, :, :, 0] * \
        grid[:, :, :, 1]  # [1, w, h, (x^2, xy)]
    basis_8[:, :, :, 0] += grid[:, :, :, 0] * \
        grid[:, :, :, 1]  # [1, w, h, (xy, y^2)]
    basis_8[:, :, :, 1] += grid[:, :, :, 1]**2  # [1, w, h, (xy, y^2)]

    flows = torch.cat(
        [names['basis_' + str(i)] for i in range(1, basis_nb + 1)], dim=0)
    if is_qr:
        # N, h, w, c --> N, h*w*c --> h*w*c, N
        flows_ = flows.view(basis_nb, -1).permute(1, 0).contiguous()
        flow_q, _ = torch.qr(flows_)
        flow_q = flow_q.permute(1, 0).reshape(basis_nb, h, w, 2).contiguous()
        flows = flow_q

    if is_scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0, 3, 1, 2).contiguous()


def subspace_project(input, vectors):
    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1).contiguous()
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2,
                                                            1)).contiguous()
    output = torch.bmm(V, project_feature).permute(0, 2,
                                                   1).view(b_, c_, h_,
                                                           w_).contiguous()

    return output


class Subspace(nn.Module):

    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):
        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):
        sub = self.Block(x)
        x = subspace_project(x, sub)

        return x


class SubspaceBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(SubspaceBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=False)

        self.conv0 = conv(inplanes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          dilation=1,
                          isReLU=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.conv1 = conv(planes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          dilation=1,
                          isReLU=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes,
                          planes,
                          kernel_size=1,
                          stride=1,
                          dilation=1,
                          isReLU=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.relu(self.bn0(self.conv0(x)))

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ShareFeature(nn.Module):

    def __init__(self, num_chs):
        super(ShareFeature, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(num_chs, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def conv(in_planes,
         out_planes,
         kernel_size=3,
         stride=1,
         dilation=1,
         isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2,
                      bias=True), nn.LeakyReLU(0.1, inplace=False))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2,
                      bias=True))


class Discriminator(nn.Module):

    def __init__(self, in_channels=1, n_classes=1):
        super(Discriminator, self).__init__()
        self.cls_head = self.cls_net(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(512,
                                   n_classes,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1,
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def cls_net(input_channels, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels * 2, 32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i],
                          channels[i + 1],
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=2,
                          bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cls_head(x)
        bs = len(x)
        x = self.conv_last(x)
        x = self.pool(x).view(bs, -1)
        return x


def training_warp(image, homo, start, patch_size):
    b, _, h, w = image.shape
    patch_size_h, patch_size_w = patch_size

    warped_image = []
    warped_img = ge.warp_perspective(image, homo.float(), dsize=(h, w))

    for b_ in range(b):
        x, y = start[b_].squeeze(-1).squeeze(-1).detach().cpu().numpy()
        x, y = int(x), int(y)
        img_ = warped_img[b_].unsqueeze(0)
        img_patch_ = img_[:, :, y:y + patch_size_h, x:x + patch_size_w]

        warped_image.append(img_patch_)

    warped_images = torch.cat(warped_image, dim=0)

    return warped_images


class Net(nn.Module):
    # 224*224
    def __init__(self, params):
        super(Net, self).__init__()

        self.params = params

        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.basis_vector_num = 16
        self.crop_size = self.params.crop_size

        self.share_feature = ShareFeature(1)
        self.conv1 = nn.Conv2d(2,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=False)
        self.block = BasicBlock
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block,
                                       128,
                                       self.layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(self.block,
                                       256,
                                       self.layers[2],
                                       stride=2)
        self.sp_layer3 = Subspace(256)
        self.layer4 = self._make_layer(self.block,
                                       512,
                                       self.layers[3],
                                       stride=2)
        self.sp_layer4 = Subspace(512)
        self.subspace_block = SubspaceBlock(2, self.basis_vector_num)

        self.conv_last = nn.Conv2d(512,
                                   8,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=8,
                                   bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data_batch):

        img1_patch, img2_patch = data_batch[
            'imgs_gray_patch'][:, :1, ...].contiguous(
            ), data_batch['imgs_gray_patch'][:, 1:2, ...].contiguous()
        img1_full, img2_full = data_batch[
            "imgs_gray_full"][:, :1, ...].contiguous(
            ), data_batch["imgs_gray_full"][:, 1:2, ...].contiguous()
        h4pt = data_batch['h4pt']

        batch_size, _, h_patch, w_patch = img1_patch.size()
        batch_size, _, h_full, w_full = img1_full.size()

        img1_patch_fea, img2_patch_fea = self.share_feature(
            img1_patch), self.share_feature(img2_patch)

        x = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x)  # bs,8,h,w
        offset_f = self.pool(x).view(batch_size, -1)  # bs,8,1,1
        Homo_f = DLT_solve(h4pt, offset_f).squeeze(1)

        x = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.sp_layer3(self.layer3(x))
        x = self.sp_layer4(self.layer4(x))
        x = self.conv_last(x)  # bs,8,h,w
        offset_b = self.pool(x).view(batch_size, -1)  # bs,8,1,1
        Homo_b = DLT_solve(h4pt, offset_b).squeeze(1)

        return {
            "Homo_b": Homo_b,
            "Homo_f": Homo_f,
            'offset_f': offset_f,
            'offset_b': offset_b
        }


class Spatial_Attention(nn.Module):

    def __init__(self, dim, num_heads, bias):
        super(Spatial_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3,
                                    dim * 3,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1,
                                    groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v,
                      'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out,
                        'b head c (h w) -> b (head c) h w',
                        head=self.num_heads,
                        h=h,
                        w=w)

        out = self.project_out(out)
        return out


class mask_predictor(nn.Module):

    def __init__(self):
        super(mask_predictor, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=2,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=2,
                               bias=False)
        self.conv1 = nn.Conv2d(in_channels=16,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=2,
                               bias=False)

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)

        self.conv4 = nn.Conv2d(16,
                               8,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               groups=8,
                               bias=False)
        self.conv5 = nn.Conv2d(8,
                               1,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)

        self.attention0 = Spatial_Attention(dim=64, num_heads=8, bias=False)
        self.attention1 = Spatial_Attention(dim=64, num_heads=8, bias=False)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)

        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out0 = self.relu(self.conv1(self.conv0(x)))
        out0_down = self.down(out0)

        out1 = self.attention1(
            self.conv3(self.conv2(self.attention0(out0_down))))
        out1_up = self.up(out1)

        mask = self.sigmoid(self.conv5(self.conv4(out1_up + out0)))
        return mask


class FlowEstimatorDense_temp(nn.Module):

    def __init__(self,
                 ch_in=64,
                 f_channels=(128, 128, 96, 64, 32, 32),
                 ch_out=2):
        super(FlowEstimatorDense_temp, self).__init__()

        N = 0
        ind = 0
        N += ch_in

        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1

        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1

        self.conv_last = conv(N, ch_out, isReLU=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        # print(x1.shape)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        # print(x2.shape)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        # print(x3.shape)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        # print(x4.shape)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        # print(x5.shape)
        x_out = self.conv_last(x5)
        # print(x_out.shape)
        x_out = self.sigmoid(x_out)
        return x_out


class FlowMaskEstimator(FlowEstimatorDense_temp):

    def __init__(self, ch_in, f_channels, ch_out):
        super(FlowMaskEstimator, self).__init__(ch_in=ch_in,
                                                f_channels=f_channels,
                                                ch_out=ch_out)


class OSNet(nn.Module):
    # 224*224
    def __init__(self,
                 params,
                 backbone,
                 init_mode="resnet",
                 norm_layer=nn.LayerNorm):
        super(OSNet, self).__init__()

        self.init_mode = init_mode
        self.params = params
        self.fea_extra = self.feature_extractor(self.params.in_channels, 1)
        # self.mask_generator = mask_predictor()
        self.mask_generator = FlowMaskEstimator(2, (8, 16, 32, 16, 8), 1)
        self.h_net = backbone(params, norm_layer=norm_layer)
        self.basis = gen_basis(self.params.crop_size[0],
                               self.params.crop_size[1]).unsqueeze(0).reshape(
                                   1, 8, -1)
        self.apply(self._init_weights)

        self.idx = 0

    def _init_weights(self, m):
        if "swin" in self.init_mode:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif "resnet" in self.init_mode:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * block.expansion,
                          1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * block.expansion))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def feature_extractor(input_channels,
                          out_channles,
                          kernel_size=3,
                          padding=1):
        layers = []
        channels = [input_channels // 2, 4, 8, out_channles]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i],
                          channels[i + 1],
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def normalize_fea(self, fea):
        _min = torch.min(fea)
        _max = torch.max(fea)
        return (fea - _min) / _max

    def forward(self, data_batch, step=0, inference=False):
        img1_patch, img2_patch = data_batch[
            'imgs_gray_patch'][:, :1,
                               ...], data_batch['imgs_gray_patch'][:, 1:2, ...]
        img1_full, img2_full = data_batch[
            "imgs_gray_full"][:, :1, ...], data_batch["imgs_gray_full"][:, 1:2,
                                                                        ...]
        img1_rgb_full, img2_rgb_full = data_batch[
            "imgs_rgb_full"][:, :3, ...], data_batch["imgs_rgb_full"][:, 3:6,
                                                                      ...]
        ganhomo_mask = data_batch["ganhomo_mask"]

        bs, _, h_patch, w_patch = img1_patch.size()
        bs, _, h_full, w_full = img1_full.size()

        # ==========================full features======================================
        img1_patch_fea, img2_patch_fea = list(
            map(self.fea_extra, [img1_patch, img2_patch]))

        # mask_f = self.mask_generator(torch.cat([img1_patch_fea, img2_patch_fea], dim=1))
        # mask_b = self.mask_generator(torch.cat([img2_patch_fea, img1_patch_fea], dim=1))

        # ========================forward ====================================

        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        weight_f = self.h_net(forward_fea)
        H_flow_f = (self.basis.to(forward_fea.device) *
                    weight_f).sum(1).reshape(bs, 2, h_patch, w_patch)

        # ========================backward===================================
        backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        weight_b = self.h_net(backward_fea)
        H_flow_b = (self.basis.to(backward_fea.device) *
                    weight_b).sum(1).reshape(bs, 2, h_patch, w_patch)

        img2_patch_fea_remap = get_warp_flow(img2_patch_fea, H_flow_f)
        img1_patch_fea_remap = get_warp_flow(img1_patch_fea, H_flow_b)
        mask_f = self.mask_generator(
            torch.cat([img1_patch_fea, img2_patch_fea_remap], dim=1))
        mask_b = self.mask_generator(
            torch.cat([img2_patch_fea, img1_patch_fea_remap], dim=1))

        mask_b_warp = torch.clamp(get_warp_flow(mask_b, H_flow_f), 0, 1)
        mask_fusion = mask_b_warp * mask_f
        # mask_fusion = mask_fusion.clip(0, 1)
        # normalize mask
        _mask_min = torch.min(mask_fusion)
        _mask_max = torch.max(torch.abs(mask_fusion))
        mask_fusion = (mask_fusion - _mask_min) / _mask_max

        img2_patch_remap = get_warp_flow(img2_patch, H_flow_f)
        # img1_patch_remap = get_warp_flow(img1_patch, H_flow_b)

        if step % 2000 == 0 and self.training:
            if not os.path.exists(f"unit_test/{self.params.exp_name}"):
                os.mkdir(f"unit_test/{self.params.exp_name}")
            buf_mask_src = torch.cat([
                mask_f,
                mask_f,
                torch.zeros_like(mask_f),
                torch.zeros_like(mask_b),
            ], -1)
            buf_img_src = torch.cat([
                img1_patch,
                img1_patch * mask_f,
                img1_patch,
                img1_patch * mask_b,
            ], -1)
            buf_fea_src = torch.cat([
                img1_patch_fea,
                img1_patch_fea * mask_f,
                img1_patch_fea,
                img1_patch_fea * mask_b,
            ], -1)
            buf_src = torch.cat([buf_mask_src, buf_img_src, buf_fea_src], -2)

            buf_mask_tar = torch.cat([
                mask_b,
                mask_b,
                torch.zeros_like(mask_f),
                torch.zeros_like(mask_b),
            ], -1)
            buf_img_tar = torch.cat([
                img2_patch,
                img2_patch * mask_f,
                img2_patch_remap,
                img2_patch_remap * mask_b,
            ], -1)
            buf_fea_tar = torch.cat([
                img2_patch_fea,
                img2_patch_fea * mask_f,
                img2_patch_fea_remap,
                img2_patch_fea_remap * mask_b,
            ], -1)
            buf_tar = torch.cat([buf_mask_tar, buf_img_tar, buf_fea_tar], -2)

            _square_bs = (math.floor(math.sqrt(buf_src[0].shape[0])))**2
            utils.save_image(
                buf_src[:_square_bs],
                f'unit_test/{self.params.exp_name}/mask_{step}_src_{torch.cuda.current_device()}.png',
                nrow=int(math.sqrt(_square_bs)))
            utils.save_image(
                buf_tar[:_square_bs],
                f'unit_test/{self.params.exp_name}/mask_{step}_tar_{torch.cuda.current_device()}.png',
                nrow=int(math.sqrt(_square_bs)))

            make_gif(
                f'unit_test/{self.params.exp_name}/mask_{step}_src_{torch.cuda.current_device()}.png',
                f'unit_test/{self.params.exp_name}/mask_{step}_tar_{torch.cuda.current_device()}.png',
                self.params.exp_name,
                f'mask_{step}_src_tar_{torch.cuda.current_device()}',
            )

        # if not self.training:
        #     if not os.path.exists(f"unit_test/testset"):
        #         os.mkdir(f"unit_test/testset")
        #     self.idx += 1

        #     buf_mask_src = torch.cat([
        #         mask_f,
        #         mask_f,
        #         torch.zeros_like(mask_f),
        #         torch.zeros_like(mask_b),
        #     ], -1)
        #     buf_img_src = torch.cat([
        #         img1_patch,
        #         img1_patch * mask_f,
        #         img1_patch,
        #         img1_patch * mask_b,
        #     ], -1)
        #     buf_fea_src = torch.cat([
        #         img1_patch_fea,
        #         img1_patch_fea * mask_f,
        #         img1_patch_fea,
        #         img1_patch_fea * mask_b,
        #     ], -1)
        #     buf_src = torch.cat([buf_mask_src, buf_img_src, buf_fea_src], -2)

        #     buf_mask_tar = torch.cat([
        #         mask_b,
        #         mask_b,
        #         torch.zeros_like(mask_f),
        #         torch.zeros_like(mask_b),
        #     ], -1)
        #     buf_img_tar = torch.cat([
        #         img2_patch,
        #         img2_patch * mask_f,
        #         img2_patch_remap,
        #         img2_patch_remap * mask_b,
        #     ], -1)
        #     buf_fea_tar = torch.cat([
        #         img2_patch_fea,
        #         img2_patch_fea * mask_f,
        #         img2_patch_fea_remap,
        #         img2_patch_fea_remap * mask_b,
        #     ], -1)
        #     buf_tar = torch.cat([buf_mask_tar, buf_img_tar, buf_fea_tar], -2)

        #     _square_bs = (math.floor(math.sqrt(buf_src[0].shape[0])))**2
        #     utils.save_image(buf_src[:_square_bs],
        #                      f'unit_test/testset/mask_{self.idx}_src_{torch.cuda.current_device()}.png',
        #                      nrow=int(math.sqrt(_square_bs)))
        #     utils.save_image(buf_tar[:_square_bs],
        #                      f'unit_test/testset/mask_{self.idx}_tar_{torch.cuda.current_device()}.png',
        #                      nrow=int(math.sqrt(_square_bs)))

        #     make_gif(
        #         f'unit_test/testset/mask_{self.idx}_src_{torch.cuda.current_device()}.png',
        #         f'unit_test/testset/mask_{self.idx}_tar_{torch.cuda.current_device()}.png',
        #         'testset',
        #         f'mask_{self.idx}_src_tar_{torch.cuda.current_device()}',
        #     )

        if not self.training:
            H_flow_f = upsample2d_flow_as(H_flow_f,
                                          img1_full,
                                          mode="bilinear",
                                          if_rate=True)
            H_flow_b = upsample2d_flow_as(H_flow_b,
                                          img1_full,
                                          mode="bilinear",
                                          if_rate=True)

            mask_f = upsample2d_flow_as(mask_f,
                                        img1_full,
                                        mode="nearest",
                                        if_rate=False)
            mask_b = upsample2d_flow_as(mask_b,
                                        img1_full,
                                        mode="nearest",
                                        if_rate=False)

            mask_b_warp = torch.clamp(get_warp_flow(mask_b, H_flow_f), 0, 1)
            mask_fusion = mask_b_warp * mask_f
            # mask_fusion = mask_fusion.clip(0, 1)
            # normalize mask
            _mask_min = torch.min(mask_fusion)
            _mask_max = torch.max(torch.abs(mask_fusion))
            mask_fusion = (mask_fusion - _mask_min) / _mask_max

            H_flow_f, H_flow_b = H_flow_f.permute(0, 2, 3,
                                                  1), H_flow_b.permute(
                                                      0, 2, 3, 1)

        fil_features = {
            "img1_patch_fea": img1_patch_fea.contiguous(),
            "img2_patch_fea": img2_patch_fea.contiguous(),
            "img1_patch_fea_warp": img1_patch_fea_remap.contiguous(),
            "img2_patch_fea_warp": img2_patch_fea_remap.contiguous(),
        }

        return {
            'img2_patch': img2_patch.contiguous(),
            'img1_full': img1_full.contiguous(),
            'img2_full': img2_full.contiguous(),
            'img1_rgb_full': img1_rgb_full.contiguous(),
            'img2_rgb_full': img2_rgb_full.contiguous(),
            'flow_f': H_flow_f.contiguous(),
            'flow_b': H_flow_b.contiguous(),
            'mask_f': mask_f.contiguous(),
            'mask_b': mask_b.contiguous(),
            'ganhomo_mask': ganhomo_mask.contiguous(),
            'fil_features': fil_features,
            'mask_fusion': mask_fusion,
        }


def Ms_Transformer(**kwargs):
    """Constructs a Multi-scale Transformer model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = OSNet(backbone=SwinTransformer, **kwargs)
    return model


def fetch_net(params):
    if params.net_type == "BasesHomo":
        model = Net(params)
    elif params.net_type == "HomoGAN":
        model = Ms_Transformer(params=params)
    else:
        raise NotImplementedError
    return model
