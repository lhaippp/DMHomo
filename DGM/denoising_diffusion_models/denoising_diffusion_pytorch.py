import os
import cv2
import pdb
import time
import math
import copy
import glob
import torch
import pickle
import imageio
import inspect

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from random import random
from torch.optim import Adam
from torch import nn, einsum
from functools import partial
from collections import namedtuple
from einops import rearrange, reduce
from multiprocessing import cpu_count
from matplotlib.colors import hsv_to_rgb
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_models.version import __version__

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1))


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim),
                                    requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(
            time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out,
                                  1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out,
                        'b h c (x y) -> b (h c) x y',
                        h=self.heads,
                        x=h,
                        y=w)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model


class Unet(nn.Module):

    def __init__(self,
                 dim,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=3,
                 self_condition=False,
                 resnet_block_groups=8,
                 learned_variance=False,
                 learned_sinusoidal_cond=False,
                 random_fourier_features=False,
                 learned_sinusoidal_dim=16):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinu_pos_emb,
                                      nn.Linear(fourier_dim, time_dim),
                                      nn.GELU(), nn.Linear(time_dim, time_dim))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                        dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim),
                    block_klass(dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    # 图解gather函数 https://zhuanlan.zhihu.com/p/352877584
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):

    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            objective='pred_noise',
            beta_schedule='cosine',
            # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_gamma=0.,
            p2_loss_weight_k=1,
            ddim_sampling_eta=1.):
        super().__init__()
        assert not (type(self) == GaussianDiffusion
                    and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v'
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        def register_buffer(name, val):
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) *
                        torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod /
                         (1 - alphas_cumprod))**-p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                 x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                x_start)

    def predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0], ),
                                   t,
                                   device=x.device,
                                   dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step',
                      total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = unnormalize_to_zero_to_one(img)
        # unnormalize flow from [0, 1] to [-1, 1]
        img[:, -2:] = img[:, -2:] * 2 - 1
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs,
                                    desc='sampling loop time step'):
            time_cond = torch.full((batch, ),
                                   time,
                                   device=device,
                                   dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, * \
                _ = self.model_predictions(
                    img, time_cond, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) /
                           (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        img = unnormalize_to_zero_to_one(img)
        # unnormalize flow from [0, 1] to [-1, 1] and rescale with 255
        img[:, -2:] = (img[:, -2:] * 2 - 1) * 512
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)),
                      desc='interpolation sample time step',
                      total=t):
            img = self.p_sample(
                img, torch.full((b, ), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        # todo: what is self-conditioning?
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)


class CifarDataset(Dataset):

    def __init__(self,
                 folder,
                 image_size,
                 exts=['jpg', 'jpeg', 'png', 'tiff'],
                 augment_horizontal_flip=False,
                 convert_image_to=None):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        self.cifar_data = [
            self.unpickle(os.path.join(folder,
                                       "data_batch_{}".format(i)))[b'data']
            for i in range(1, 6)
        ]
        self.cifar_data = np.concatenate(self.cifar_data, 0)

        self.datas = self.cifar_data.reshape(-1, 3, 32,
                                             32).transpose(0, 2, 3, 1)
        # print("cifar data shape is {}".format(self.datas.shape))
        # for i in range(10):
        #     import cv2
        #     cv2.imwrite("/data/denoising-diffusion-pytorch/SIGNS_dataset/test_{}.png".format(i), self.datas[i])

        maybe_convert_fn = partial(
            convert_image_to_fn,
            convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip()
            if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict

    def __len__(self):
        # print("cifar dataset contains {} items".format(len(self.datas)))
        return len(self.datas)

    def __getitem__(self, index):
        img = self.datas[index]
        img = Image.fromarray(np.uint8(img))
        return self.transform(img)


class GHOFTestDataset(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        super().__init__()

        self.samples = np.load(benchmark_path, allow_pickle=True)

        self.image_size = image_size

        maybe_convert_fn = partial(
            convert_image_to_fn,
            convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            # T.Resize(image_size),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip()
            if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]["img1"]
        img = Image.fromarray(np.uint8(img))
        return self.transform(img)


def imdecode(data, require_chl3=True, require_alpha=False):
    img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_UNCHANGED)

    assert img is not None, 'failed to decode'
    if img.ndim == 2 and require_chl3:
        img = img.reshape(img.shape + (1, ))
    if img.shape[2] == 1 and require_chl3:
        img = np.tile(img, (1, 1, 3))
    if img.ndim == 3 and img.shape[2] == 3 and require_alpha:
        assert img.dtype == np.uint8
        img = np.concatenate([img, np.ones_like(img[:, :, :1]) * 255], axis=2)
    return img


class HomoTrainData(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):

        self.data_infor = open(benchmark_path, 'r').readlines()

        self.image_size = image_size

        maybe_convert_fn = partial(
            convert_image_to_fn,
            convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            # T.RandomCrop(image_size),
            T.RandomHorizontalFlip()
            if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        # return size of dataset
        return len(self.data_infor)

    def __getitem__(self, idx):

        # img loading
        img_names = self.data_infor[idx]
        img_names = img_names.split(' ')

        data1 = self.nf.get(img_names[0])  # Read image according to data list
        data2 = self.nf.get(img_names[1][:-1])

        img1 = imdecode(data1)
        img2 = imdecode(data2)

        img = img1 if random() <= 0.5 else img2

        img = Image.fromarray(np.uint8(img1))
        return self.transform(img).float()


def mesh_grid_np(B, H, W):
    # mesh grid
    x_base = np.arange(0, W)
    x_base = np.tile(x_base, (B, H, 1))

    y_base = np.arange(0, H)  # BHW
    y_base = np.tile(y_base, (B, W, 1)).transpose(0, 2, 1)

    ones = np.ones_like(x_base)

    base_grid = np.stack([x_base, y_base, ones], 1)  # B3HW
    return base_grid


def get_flow_np(H_mat_mul, patch_indices, image_size_h=600, image_size_w=800):
    # (N, 6, 3, 3)
    batch_size = H_mat_mul.shape[0]
    divide = H_mat_mul.shape[1]
    H_mat_mul = H_mat_mul.reshape(batch_size, divide, 3, 3)

    small_patch_sz = [image_size_h // divide, image_size_w]
    small = 1e-7

    H_mat_pool = np.zeros((batch_size, image_size_h, image_size_w, 3, 3))

    for i in range(divide):
        H_mat = H_mat_mul[:, i, :, :]

        if i == divide - 1:
            H_mat = np.broadcast_to(
                np.expand_dims(np.expand_dims(H_mat, 1), 1),
                (batch_size, image_size_h - i * small_patch_sz[0],
                 image_size_w, 3, 3))
            H_mat_pool[:, i * small_patch_sz[0]:, ...] = H_mat
            continue

        H_mat = np.broadcast_to(np.expand_dims(np.expand_dims(
            H_mat, 1), 1), (batch_size, small_patch_sz[0], image_size_w, 3, 3))
        H_mat_pool[:, i * small_patch_sz[0]:(i + 1) * small_patch_sz[0],
                   ...] = H_mat

    pred_I2_index_warp = np.expand_dims(patch_indices.transpose(0, 2, 3, 1), 4)
    pred_I2_index_warp = np.matmul(
        H_mat_pool, pred_I2_index_warp)[:, :, :, :, 0].transpose(0, 3, 1, 2)
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    warp_index = np.concatenate((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = warp_index - vgrid
    # NCHW to HWC
    return flow.squeeze().transpose(1, 2, 0)


def homo_to_flow(homo, H=600, W=800):
    img_indices = mesh_grid_np(B=1, H=H, W=W)
    flow_gyro = get_flow_np(homo, img_indices, image_size_h=H, image_size_w=W)
    return flow_gyro.astype(np.float32)


def adapt_homography_to_preprocessing_v3(h0, w0, H, h1, w1):
    M_0 = np.array([[w0 / 2.0, 0., w0 / 2.0], [0., h0 / 2.0, h0 / 2.0],
                    [0., 0., 1.]])
    M_0_inv = np.linalg.inv(M_0)
    H_0_norm = np.matmul(np.matmul(M_0_inv, H), M_0)

    M_1 = np.array([[w1 / 2.0, 0., w1 / 2.0], [0., h1 / 2.0, h1 / 2.0],
                    [0., 0., 1.]])
    M_1_inv = np.linalg.inv(M_1)
    H_1 = np.matmul(np.matmul(M_1, H_0_norm), M_1_inv)
    return H_1


RE = [
    '000004', '000008', '000009', '0000010', '0000012', '0000013', '0000014',
    '0000015', '0000017', '0000018', '0000052', '0000053', '0000054',
    '0000055', '0000065', '0000066', '0000068', '0000070', '00000105',
    '00000108', '00000111', '00000112', '00000113', '00000114', '00000116',
    '00000120', '00000122', '00000121', '00000125', '00000126', '00000127',
    '00000128', '00000130', '00000131', '00000132', '00000133', '00000134',
    '00000136', '00000138', '00000142', '00000143', '00000144', '00000145',
    '00000151', '00000153', '00000154', '00000156', '00000157', '00000159',
    '00000160', '00000162', '00000167', '00000168', '00000203', '00000204',
    '00000205', '00000206', '00000207', '00000208', '00000209', '00000212',
    '00000231', '00000233', '00000234'
]

LT = [
    '0000032', '0000033', '0000036', '0000037', '0000039', '0000040',
    '0000041', '0000042', '0000043', '0000045', '0000048', '0000049',
    '0000050', '0000051', '00000123', '00000150', '00000175', '00000176',
    '00000178', '00000179', '00000180', '00000182', '00000183', '00000184',
    '00000186', '00000187', '00000189', '00000237', '00000240', '00000245',
    '00000246'
]

LL = [
    '0000071', '0000072', '0000073', '0000074', '0000075', '0000076',
    '0000077', '0000078', '0000079', '0000080', '0000081', '0000082',
    '0000083', '0000084', '0000086', '0000087', '0000088', '0000089',
    '0000090', '0000093', '0000094', '0000095', '0000096', '0000097',
    '0000098', '0000099', '00000214', '00000215', '00000217', '00000218',
    '00000219', '00000220', '00000221', '00000222', '00000223', '00000224',
    '00000225', '00000227', '00000228'
]

SF = [
    '000001', '000002', '000003', '000007', '0000057', '0000058', '0000059',
    '0000060', '0000061', '0000062', '0000063', '0000067', '0000069',
    '00000101', '00000102', '00000103', '00000106', '00000170', '00000171',
    '00000172', '00000173', '00000174', '00000185', '00000190', '00000191',
    '00000192', '00000193', '00000202', '00000210', '00000211', '00000213',
    '00000229', '00000230', '00000235', '00000236', '00000241', '00000242',
    '00000243', '00000247', '00000248', '00000249', '00000250'
]

LF = [
    '000005', '000006', '0000019', '0000020', '0000021', '0000022', '0000023',
    '0000024', '0000025', '0000027', '0000028', '0000029', '0000056',
    '0000064', '00000109', '00000110', '00000117', '00000118', '00000119',
    '00000124', '00000135', '00000137', '00000139', '00000140', '00000146',
    '00000148', '00000149', '00000152', '00000161', '00000163', '00000164',
    '00000165', '00000166', '00000169', '00000194', '00000195', '00000196',
    '00000197', '00000198', '00000199', '00000201', '00000232'
]


class PseudoCondition(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
        total_data_slice_idx=1,
        data_slice_idx=0,
        isGenerate=False,
    ):
        # assert phase in ['train', 'val', 'test']
        self.cond_npys = sorted(glob.glob(os.path.join(benchmark_path,
                                                       '*npy')))
        slice_cond_length = (len(self.cond_npys) // total_data_slice_idx)
        # slice the whole dataset into sub-sets to sample on multiple GPUs
        self.cond_npys = self.cond_npys[data_slice_idx *
                                        slice_cond_length:(data_slice_idx +
                                                           1) *
                                        slice_cond_length]

        self.image_size = image_size

        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
        ])

        # for unit test
        self.cnt = 0

        self.isGenerate = isGenerate

        print(
            f"the first npy pth is {self.cond_npys[0]}, be sure to check they are different on each GPU"
        )
        time.sleep(10)

    def __len__(self):
        print(f"The length of UnHomoTrainData is {len(self.cond_npys)}")
        return len(self.cond_npys)

    def __getitem__(self, idx):
        # img loading
        # data: [mask, (homo_forward, homo_backward)]
        data = np.load(self.cond_npys[idx], allow_pickle=True).item()

        homof, homob = data['homo'][0].squeeze(), data['homo'][1].squeeze()

        ganhomo_mask = data['mask']

        scene_class = 0

        motionf = homo_to_flow(homof[None, None], self.image_size,
                               self.image_size)
        # we need to convert flow field into RGB flow, details plz see ablation study
        rgb_homoflow_forward = flow_to_image(motionf)

        img = np.concatenate((ganhomo_mask, rgb_homoflow_forward, motionf),
                             axis=2)
        return self.transform(img).float(), scene_class


class CATestSet(Dataset):

    def __init__(
        self,
        benchmark_path,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip=False,
        convert_image_to=None,
    ):
        # 路径
        # files_path = '/data/for_jr/' CVPR2021_list pair_path_ECCV
        self.npy_list = os.path.join(benchmark_path, "test.txt")
        self.npy_path = os.path.join(benchmark_path, "pt/")  # NPYFile Npz_Set
        self.image_path = os.path.join(benchmark_path, "img/")

        self.data_infor = open(self.npy_list, 'r').readlines()

        self.image_size = image_size

        # maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            # T.CenterCrop(image_size),
        ])

    def __len__(self):
        # return size of dataset
        # print(f"dataset length {len(self.data_infor)}")
        return len(self.data_infor)

    def points2homo(self, pt_set):
        src = []
        dst = []
        for j in range(6):
            src.append(pt_set[j][0])
            dst.append(pt_set[j][1])

        src = np.reshape(src, (1, -1, 2)).astype('float32')
        dst = np.reshape(dst, (1, -1, 2)).astype('float32')

        homo, _ = cv2.findHomography(src, dst)
        return homo

    def __getitem__(self, idx):
        img_names = self.data_infor[idx].replace('\n', '')

        video_names = img_names.split('/')[0]
        img_names = img_names.split(' ')

        npy_name = img_names[0].split('/')[-1] + '_' + img_names[1].split(
            '/')[-1] + '.npy'

        save_name = img_names[0].split('.')[0].split(
            '/')[1] + '_' + img_names[1].split('.')[0].split('/')[1]

        img1 = cv2.imread(os.path.join(self.image_path, img_names[0])) / 255.
        img2 = cv2.imread(os.path.join(self.image_path, img_names[1])) / 255.

        img1 = cv2.resize(img1, (self.image_size, self.image_size))
        img2 = cv2.resize(img2, (self.image_size, self.image_size))

        correspendeces = np.load(os.path.join(self.npy_path, npy_name),
                                 allow_pickle=True).item()['matche_pts']

        homo = self.points2homo(correspendeces)

        homo = adapt_homography_to_preprocessing_v3(360, 640, homo,
                                                    self.image_size,
                                                    self.image_size)
        motion = homo_to_flow(homo[None, None], self.image_size,
                              self.image_size)

        rgb_homoflow = flow_to_image(motion)

        img = np.concatenate((img1, img2, rgb_homoflow, motion), axis=2)
        return self.transform(img).float()


def resize_flow(flow, size):
    h, w, _ = flow.shape

    res = cv2.resize(flow, (size, size))

    u_scale = (size / w)
    v_scale = (size / h)

    res[:, :, 0] = res[:, :, 0] * u_scale
    res[:, :, 1] = res[:, :, 1] * v_scale
    return res


def flow_warp(x, flow12, pad="border", mode="bilinear"):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if "align_corners" in inspect.getfullargspec(
            torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x,
                                               v_grid,
                                               mode=mode,
                                               padding_mode=pad,
                                               align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x,
                                               v_grid,
                                               mode=mode,
                                               padding_mode=pad)
    return im1_recons


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_to_image_luo(flow, display=False):
    """

        :param flow: H,W,2
        :param display:
        :return: H,W,3
        """

    def compute_color(u, v):

        def make_color_wheel():
            """
                Generate color wheel according Middlebury color code
                :return: Color wheel
                """
            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3])

            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY,
                       1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
            col += RY

            # YG
            colorwheel[col:col + YG, 0] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, YG) / YG))
            colorwheel[col:col + YG, 1] = 255
            col += YG

            # GC
            colorwheel[col:col + GC, 1] = 255
            colorwheel[col:col + GC,
                       2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
            col += GC

            # CB
            colorwheel[col:col + CB, 1] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, CB) / CB))
            colorwheel[col:col + CB, 2] = 255
            col += CB

            # BM
            colorwheel[col:col + BM, 2] = 255
            colorwheel[col:col + BM,
                       0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
            col += +BM

            # MR
            colorwheel[col:col + MR, 2] = 255 - \
                np.transpose(np.floor(255 * np.arange(0, MR) / MR))
            colorwheel[col:col + MR, 0] = 255

            return colorwheel

        """
            compute optical flow color map
            :param u: optical flow horizontal map
            :param v: optical flow vertical map
            :return: optical flow in color code
            """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    UNKNOWN_FLOW_THRESH = 1e7
    """
        Convert flow into middlebury color code image
        :param flow: optical flow map
        :return: optical flow image in middlebury color
        """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    if display:
        print(
            "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" %
            (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    # _min, _mean, _max = np.min(img), np.mean(img), np.max(img)
    # print(_min, _mean, _max)

    return img / 255.


def flow_to_image(flow, max_flow=256):
    # flow shape (H, W, C)
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return im


def visulize_flow(all_images):
    np_flow = all_images.detach().cpu().numpy().transpose([0, 2, 3, 1])

    vis_flow = []

    for _, flow in enumerate(np_flow):
        vis_flow.append(flow_to_image(flow))

    vis_flow_np = np.array(vis_flow, dtype=np.float32)
    vis_flow_np = vis_flow_np.transpose([0, 3, 1, 2])

    vis_flow_torch = torch.from_numpy(vis_flow_np)
    # print("vis_flow_torch shape ", vis_flow_torch.shape)
    return vis_flow_torch


def postProcess(torch_tensor, all_ganHomomask, flows):
    img1s = torch_tensor[:, :3]
    img2s = torch_tensor[:, 3:6]

    warp_img2s = flow_warp(img2s, flows)

    flows_vis = visulize_flow(flows).cuda(flows.device)

    all_ganHomomask_rgb = all_ganHomomask.repeat(1, 3, 1, 1)

    buf1 = torch.concat([img1s, img1s, all_ganHomomask_rgb, flows_vis], -1)
    buf2 = torch.concat([img2s, warp_img2s, all_ganHomomask_rgb, flows_vis],
                        -1)
    return buf1, buf2


def postProcess_cv2(imgs, homos, rank):
    img1s = imgs[:, :3].astype(np.float32) / 255.
    img2s = imgs[:, 3:6].astype(np.float32) / 255.

    warp_img2s = []
    for i, img in enumerate(img1s):
        img = img.transpose(1, 2, 0)
        warp_img = cv2.warpPerspective(img, homos[i], (256, 256))
        warp_img = warp_img.transpose(2, 0, 1)[None]
        warp_img2s.append(warp_img)

    warp_img2s = np.concatenate(warp_img2s, 0)
    # print(f"warp_img2s shape {warp_img2s.shape}")

    img1s = torch.from_numpy(img1s).cuda(rank)
    img2s = torch.from_numpy(img2s).cuda(rank)
    warp_img2s = torch.from_numpy(warp_img2s).cuda(rank)

    buf1 = torch.concat([img1s, warp_img2s], -1)
    buf2 = torch.concat([img2s, img2s], -1)
    return buf1, buf2


def make_gif(img1, img2, name):
    img1, img2 = cv2.imread(img1), cv2.imread(img2)
    with imageio.get_writer(f'sample_gif_results/{name}.gif',
                            mode='I',
                            duration=0.5) as writer:
        writer.append_data(img1[:, :, ::-1])
        writer.append_data(img2[:, :, ::-1])


def get_grid(batch_size, H, W, start=0):

    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
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


def homo_gen(flow):
    b, c, h, w = flow.shape

    grid = get_grid(b, h, w)

    if flow.is_cuda:
        grid = grid.cuda()

    grid = grid.permute(0, 2, 3, 1).reshape(b, 1, -1, 2).type(torch.float64)

    flow = flow.permute(0, 2, 3, 1).reshape(b, 1, -1, 2)

    homo = DLT_solve(grid, flow)

    return homo


def saveTrainPair(torch_tensor, ganHomo_mask, flows):
    assert torch.max(
        torch_tensor
    ) <= 1, f"image should be normalized to [0, 1], not[{torch.min(torch_tensor)}, {torch.max(torch_tensor)}]"
    # ret = torch.cat([torch_tensor, flows, mask], axis=1)
    # assert ret.shape[1] == 9, f"ret dimension should be equal to 9, not {ret.shape[1]}"

    imgs_np = torch_tensor.detach().cpu().numpy()
    # convert imgs from float32 to uint8
    imgs_np = (imgs_np * 255).astype(np.uint8)

    homos = homo_gen(flows)
    homos = homos.detach().cpu().numpy().squeeze()
    # print(f"imgs_np shape {imgs_np.shape} | homos shape {homos.shape}")
    return {
        "imgs": imgs_np,
        "homos": homos,
    }


class Trainer(object):

    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=9,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        num_worker=8,
        total_data_slice_idx=1,
        data_slice_idx=0,
        shuffle=True,
        isGenerate=False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        self.ds = PseudoCondition(
            folder,
            self.image_size,
            augment_horizontal_flip=augment_horizontal_flip,
            convert_image_to=convert_image_to,
            total_data_slice_idx=total_data_slice_idx,
            data_slice_idx=data_slice_idx,
            isGenerate=isGenerate,
        )

        dl = DataLoader(self.ds,
                        batch_size=train_batch_size,
                        shuffle=shuffle,
                        pin_memory=True,
                        num_workers=num_worker)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr,
                        betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model,
                           beta=ema_decay,
                           update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step':
            self.step,
            'model':
            self.accelerator.get_state_dict(self.model),
            'opt':
            self.opt.state_dict(),
            'ema':
            self.ema.state_dict(),
            'scaler':
            self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler) else None,
            'version':
            __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        if not self.accelerator.is_main_process:
            return

        data = torch.load(str(milestone), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step,
                  total=self.train_num_steps,
                  disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    with self.accelerator.autocast():
                        loss = self.model(data[0].to(device),
                                          classes=data[1].to(device))
                        # if torch.isnan(loss):
                        #     continue
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples,
                                                    self.batch_size)
                            # specifiy a certain flow
                            # data[0]: (img1, img2, ganhomo_mask, rgb_homoflow_forward, motionf, mask_fusion)
                            # data[0] channels: (3, 3, 3, 3, 2, 3)
                            # print(f"data[0] shape: {data[0].shape}")
                            ganHomo_mask = data[0][:, 6:7].repeat(
                                self.num_samples, 1, 1, 1)[:self.num_samples]
                            rgb_flows = data[0][:, 9:12].repeat(
                                self.num_samples, 1, 1, 1)[:self.num_samples]
                            flows = data[0][:, 12:14].repeat(
                                self.num_samples, 1, 1, 1)[:self.num_samples]
                            dmHomo_mask = data[0][:, 14:15].repeat(
                                self.num_samples, 1, 1, 1)[:self.num_samples]
                            all_images_list = list(
                                map(
                                    lambda n: self.ema.ema_model.sample(
                                        classes=torch.randint(0, 1,
                                                              (n, )).cuda(),
                                        rgb_flow=rgb_flows[:n, ],
                                        flow=flows[:n, ],
                                        ganHomo_mask=ganHomo_mask[:n, ],
                                        dmHomo_mask=dmHomo_mask[:n, ],
                                    ), batches))

                        all_images, all_ganHomomask, all_flows, all_dmHomomask = [], [], [], []
                        for buf in all_images_list:
                            all_images.append(buf[0])
                            all_ganHomomask.append(buf[1])
                            all_flows.append(buf[2])
                            all_dmHomomask.append(buf[3])

                        all_images = torch.cat(all_images, dim=0)
                        all_ganHomomask = torch.cat(all_ganHomomask, dim=0)
                        all_flows = torch.cat(all_flows, dim=0)
                        all_dmHomomask = torch.cat(all_dmHomomask, dim=0)

                        img1s, warp_img2s = postProcess(
                            all_images,
                            all_ganHomomask=all_ganHomomask,
                            flows=all_flows,
                            all_dmHomomask=all_dmHomomask)

                        # from rgb to bgr
                        permute = [2, 1, 0]
                        img1s, warp_img2s = img1s[:,
                                                  permute], warp_img2s[:,
                                                                       permute]

                        utils.save_image(img1s,
                                         str(self.results_folder /
                                             f'sample-{milestone}-source.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        utils.save_image(warp_img2s,
                                         str(self.results_folder /
                                             f'sample-{milestone}-target.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        make_gif(
                            str(self.results_folder /
                                f'sample-{milestone}-source.png'),
                            str(self.results_folder /
                                f'sample-{milestone}-target.png'), milestone)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def sample(self, idx, rank, step=1):
        self.ema.ema_model.eval()

        data = next(self.dl)

        # specifiy a certain flow
        # data[0]: (ganhomo_mask, rgb_homoflow_forward, motionf)
        # data[0] channels: (3, 3, 2)
        ganHomo_mask = data[0][:, :1].to(rank)
        rgb_flows = data[0][:, 3:6].to(rank)
        flows = data[0][:, 6:].to(rank)

        with torch.no_grad():
            all_images = self.ema.ema_model.sample(
                # classes=torch.randint(0, 5, (flows.shape[0], )).cuda(rank),
                classes=data[1].cuda(rank),
                rgb_flow=rgb_flows,
                flow=flows,
                ganHomo_mask=ganHomo_mask,
            )

        # img, ganHomo_mask, flow, dmHomo_mask
        ret = saveTrainPair(
            all_images[0],
            ganHomo_mask=all_images[1],
            flows=all_images[2],
        )

        if step % 100 == 0:
            _square_bs = math.floor(math.sqrt(all_images[0].shape[0]))
            _square_bs = 4 if _square_bs > 4 else _square_bs
            _square_bs = _square_bs * _square_bs

            img1s, warp_img2s = postProcess(
                all_images[0][:_square_bs],
                all_ganHomomask=all_images[1][:_square_bs],
                flows=all_images[2][:_square_bs],
            )
            permute = [2, 1, 0]
            img1s, warp_img2s = img1s[:, permute], warp_img2s[:, permute]

            if not os.path.exists('generate_training_pairs'):
                os.mkdir('generate_training_pairs')

            utils.save_image(
                img1s,
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-source_flowRemap.png',
                nrow=int(math.sqrt(_square_bs)))
            utils.save_image(
                warp_img2s,
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-target_flowRemap.png',
                nrow=int(math.sqrt(_square_bs)))

            if not os.path.exists('sample_gif_results'):
                os.mkdir('sample_gif_results')

            make_gif(
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-source_flowRemap.png',
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-target_flowRemap.png',
                f'idx_{idx}_step_{step}_rank_{rank}_flowRemap',
            )

            img1s, warp_img2s = postProcess_cv2(
                imgs=ret["imgs"][:_square_bs],
                homos=ret["homos"][:_square_bs],
                rank=rank,
            )

            img1s, warp_img2s = img1s[:, permute], warp_img2s[:, permute]

            utils.save_image(
                img1s,
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-source_homoWarp.png',
                nrow=int(math.sqrt(_square_bs)))
            utils.save_image(
                warp_img2s,
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-target_homoWarp.png',
                nrow=int(math.sqrt(_square_bs)))
            make_gif(
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-source_homoWarp.png',
                f'generate_training_pairs/idx_{idx}_step_{step}_rank_{rank}_sample-target_homoWarp.png',
                f'idx_{idx}_step_{step}_rank_{rank}_homoWarp',
            )

        return ret

    def generate_test_samples(self):
        img1_ls, img2_ls, img1_img2_ls = [], [], []
        with tqdm(total=len(self.ca_test_dl)) as t:
            for batch in self.ca_test_dl:
                img1_ls.append(batch[:, :3])
                # t.set_description(desc=print_str)
                t.update()

        return torch.cat(img1_ls)
