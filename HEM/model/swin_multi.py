import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .utils import gen_basis, get_warp_flow


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 param,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 *args,
                 **kwargs):
        super().__init__()

        basis = gen_basis(
            param.crop_size[0], param.crop_size[1]).unsqueeze(0).reshape(
                1, param.num_basis,
                -1)  # self.num_basis,2,h,w --> 1, self.num_basis, 2*h*w
        self.register_buffer("basis", basis)

        self.num_basis = param.num_basis

        self.num_layers = len(param.depths)
        self.num_decoder_layers = param.num_decoder_layers
        self.embed_dim = param.embed_dim
        self.ape = param.ape
        self.patch_norm = param.patch_norm
        self.num_features = int(self.embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = param.mlp_ratio
        self.drop_path = 0
        self.activation = nn.GELU

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=param.crop_size,
            patch_size=param.patch_size,
            in_chans=param.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            activation=self.activation)
        num_patches = self.patch_embed.num_patches  # patch_number_h * patch_number_q
        self.patches_resolution = self.patch_embed.patches_resolution  # (patch_number_h, patch_number_q)

        self.query_token = nn.Parameter(
            torch.zeros(1, param.num_basis, self.num_features))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=param.drop_rate)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, param.drop_path_rate, sum(param.depths))
        ]  # stochastic depth decay rule

        # build feature extractors
        self.feature_pyramid_extractor = FeatureExtractor(
            self.embed_dim // 2, self.num_layers, self.activation)

        # build layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                input_resolution=(param.crop_size[0] // (2**(i_layer + 1)),
                                  param.crop_size[1] // (2**(i_layer + 1))),
                depth=param.depths[i_layer],
                layer_depth=param.layer_depth[i_layer],
                num_heads=param.num_heads[i_layer],
                window_size=param.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=param.qkv_bias,
                qk_scale=param.qk_scale,
                drop=param.drop_rate,
                attn_drop=param.attn_drop_rate,
                drop_path=dpr[sum(param.depths[:i_layer]
                                  ):sum(param.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging,
                use_checkpoint=use_checkpoint)
            self.encoder_layers.append(layer)

        # build blocks
        blks_list = list(range(1, self.num_layers))
        blks_list.append(self.num_layers - 1)

        self.blocks_token_only = nn.ModuleList([
            LayerScale_Block_CA(dim=self.num_features,
                                out_dim=self.num_features,
                                num_heads=param.num_heads[i],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=param.qkv_bias,
                                qk_scale=param.qk_scale,
                                drop=param.drop_rate,
                                attn_drop=param.attn_drop_rate,
                                drop_path=0.0,
                                norm_layer=norm_layer,
                                act_layer=self.activation,
                                Attention_block=Class_Attention,
                                Mlp_block=Mlp) for i in blks_list
        ])

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head1 = nn.Linear(self.num_features, self.num_features)
        self.head2 = nn.Linear(self.num_features, 1)  # self.num_basis

        self.activate = self.activation()

        trunc_normal_(self.query_token, std=.02)

    def forward(self, x):
        """
            x shape: bs, 2, h, w
        """
        # forward_features
        bs, _, h_patch, w_patch = x.shape
        query_token = self.query_token.repeat(bs, 1, 1)
        x1_patch, x2_patch = x[:, :1], x[:, 1:]
        x1_pyramid = self.feature_pyramid_extractor(x1_patch)  # + [x1_patch]
        x2_pyramid = self.feature_pyramid_extractor(x2_patch)  # + [x2_patch]
        weight_f = 0
        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            _, _, h_x, w_x, = x1.shape

            # warping
            if l == 0:
                x1_warp, x2_warp = x1, x2
            else:
                H_flow_f = (self.basis * weight_f).sum(1).reshape(
                    bs, 2, h_patch, w_patch)
                H_flow_f = upsample2d_flow_as(H_flow_f, x1, if_scale=True)
                x2_warp = get_warp_flow(x2, H_flow_f)

            x = torch.cat((x1, x2_warp), dim=1)
            x = x.flatten(2).transpose(1, 2)

            x = self.encoder_layers[self.num_layers - l - 1](x)

            query_token = self.blocks_token_only[self.num_layers - l - 1](
                query_token, x)
            query_token = self.norm(query_token)  # B L C
            # regression
            h = self.activate(self.head1(query_token))
            h = self.head2(h)

            scale = h_patch // h_x
            weight_f += h * scale

        return weight_f


class FeatureExtractor(nn.Module):

    def __init__(self, embed_dim, num_layers, activation):
        super(FeatureExtractor, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.activation = activation
        self.convs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            in_channel = int((self.embed_dim * 2**(i_layer - 1))**np.heaviside(
                i_layer, 0))  # 1, embed_dim, 2*embed_dim ...
            out_channel = int((self.embed_dim * 2**i_layer)**np.heaviside(
                i_layer + 1, 0))  # embed_dim, 2*embed_dim, 4*embed_dim...
            layer = nn.Sequential(
                nn.Conv2d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=3,
                          stride=2,
                          padding=1), self.activation(),
                nn.Conv2d(in_channels=out_channel,
                          out_channels=out_channel,
                          kernel_size=3,
                          padding=1), self.activation())
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        #self.reduce_dim = nn.Linear(3080, 1)
        self.info_flatten = nn.Conv2d(3080, 1, 1)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, :8]).reshape(B, 8, self.num_heads,
                                     C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        #k = self.info_flatten(k).permute(0, 2, 1, 3)
        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 8, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self,
                 dim,
                 out_dim,
                 num_heads=3,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Attention_block=None,
                 Mlp_block=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp_block(in_features=dim,
                              hidden_features=mlp_hidden_dim,
                              act_layer=act_layer,
                              drop=drop)  #
        self.mlp2 = Mlp_block(in_features=dim,
                              hidden_features=mlp_hidden_dim,
                              out_features=out_dim,
                              act_layer=act_layer,
                              drop=drop)
        self.norm3 = norm_layer(dim)
        init_values1 = 1e-5 if dim <= 24 else 1e-6
        init_values2 = 1e-5 if dim <= 24 else 1e-6
        self.gamma_1 = nn.Parameter(init_values1 * torch.ones((dim)),
                                    requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values2 * torch.ones((dim)),
                                    requires_grad=True)

    def forward(self, x_cls, x):
        u = torch.cat((x_cls, x), dim=1)

        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))

        x_cls = x_cls + self.drop_path(
            self.gamma_2 * self.mlp1(self.norm2(x_cls)))

        x_cls = self.mlp2(self.norm3(x_cls))
        return x_cls


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        h_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 h_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH 2*(Wh*Ww-1)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # N = Wh*Ww
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)  # 3, B_, self.num_heads, N, C // self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        self_attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        self_attn = self_attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            self_attn = self_attn.view(B_ // nW, nW, self.num_heads, N,
                                       N) + mask.unsqueeze(1).unsqueeze(0)
            self_attn = self_attn.view(-1, self.num_heads, N, N)  # N = Wh*Ww
            self_attn = self.softmax(self_attn)

        else:
            self_attn = self.softmax(self_attn)

        self_attn = self.attn_drop(self_attn)

        x = (self_attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(self.dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size,
                              None))  # shift_size = window_size // 2
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size:{}!={}*{}".format(
            L, H, W)

        shortcut = x

        x = self.norm1(x)

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)  # x'= x + SA(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # x'= x + FFN(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class WindowCrossAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        h_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 h_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH 2*(Wh*Ww-1)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x1.shape  # N = Wh*Ww
        qkv1 = self.qkv(x1).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)  # 3, B_, self.num_heads, N, C // self.num_heads
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[
            2]  # make torchscript happy (cannot use tensor as tuple)
        qkv2 = self.qkv(x2).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4)  # 3, B_, self.num_heads, N, C // self.num_heads
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        q1 = q1 * self.scale
        cross_attn = (q1 @ k2.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        cross_attn = cross_attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            cross_attn = cross_attn.view(B_ // nW, nW, self.num_heads, N,
                                         N) + mask.unsqueeze(1).unsqueeze(0)
            cross_attn = cross_attn.view(-1, self.num_heads, N, N)  # N = Wh*Ww
            cross_attn = self.softmax(cross_attn)

        else:
            cross_attn = self.softmax(cross_attn)

        cross_attn = self.attn_drop(cross_attn)

        x1 = (cross_attn @ v2).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        return x1


class SwinCrossBlock(nn.Module):
    r""" Swin Cross Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowCrossAttention(dim,
                                         window_size=to_2tuple(
                                             self.window_size),
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         attn_drop=attn_drop,
                                         proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size,
                                                       None))  #
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size,
                              None))  # shift_size = window_size // 2
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x1, x2):
        H, W = self.input_resolution
        B, L, C = x1.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x1
        shortcut2 = x2

        x1 = self.norm1(x1)
        x2 = self.norm1(x2)
        x1 = x1.view(B, H, W, C)
        x2 = x2.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1,
                                    shifts=(-self.shift_size,
                                            -self.shift_size),
                                    dims=(1, 2))
            shifted_x2 = torch.roll(x2,
                                    shifts=(-self.shift_size,
                                            -self.shift_size),
                                    dims=(1, 2))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        # partition windows
        x1_windows = window_partition(
            shifted_x1, self.window_size)  # nW*B, window_size, window_size, C
        x1_windows = x1_windows.view(-1, self.window_size * self.window_size,
                                     C)  # nW*B, window_size*window_size, C
        x2_windows = window_partition(
            shifted_x2, self.window_size)  # nW*B, window_size, window_size, C
        x2_windows = x2_windows.view(-1, self.window_size * self.window_size,
                                     C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn1_windows = self.attn(
            x1_windows, x2_windows,
            mask=self.attn_mask)  # nW*B, window_size*window_size, C
        attn2_windows = self.attn(x2_windows, x1_windows, mask=self.attn_mask)
        # merge windows
        attn1_windows = attn1_windows.view(-1, self.window_size,
                                           self.window_size, C)
        shifted_x1 = window_reverse(attn1_windows, self.window_size, H, W)
        attn2_windows = attn2_windows.view(-1, self.window_size,
                                           self.window_size, C)
        shifted_x2 = window_reverse(attn2_windows, self.window_size, H, W)
        # reverse cyclic shift
        if self.shift_size > 0:
            x1 = torch.roll(shifted_x1,
                            shifts=(self.shift_size, self.shift_size),
                            dims=(1, 2))
            x2 = torch.roll(shifted_x2,
                            shifts=(self.shift_size, self.shift_size),
                            dims=(1, 2))
        else:
            x1 = shifted_x1
            x2 = shifted_x2

        x1 = x1.view(B, H * W, C)
        x2 = x2.view(B, H * W, C)
        # FFN
        x1 = shortcut1 + self.drop_path(x1)
        x1 = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        x2 = shortcut2 + self.drop_path(x2)
        x2 = x2 + self.drop_path(self.mlp(self.norm2(x2)))

        return x1, x2


class PatchMerging_ori(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # patch_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # patch_resolution
        self.dim = dim  # 通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=2 * dim,
                               kernel_size=3,
                               bias=False,
                               stride=2,
                               padding=1)
        self.norm1 = nn.BatchNorm2d(2 * dim)
        self.conv2 = nn.Conv2d(in_channels=2 * dim,
                               out_channels=2 * dim,
                               kernel_size=3,
                               bias=False,
                               padding=1)
        self.norm2 = nn.BatchNorm2d(2 * dim)
        self.activate = nn.LeakyReLU(inplace=True)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.norm1(self.conv1(x))
        x = self.activate(x)
        x = self.norm2(self.conv2(x))
        x = self.activate(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, 2 * C)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 layer_depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build layer
        layer_list = []
        for l in range(layer_depth):
            input_resolution_current = (input_resolution[0] // 2**l,
                                        input_resolution[1] // 2**l)
            dim_current = dim * 2**l
            blocks_list = [
                SwinTransformerBlock(dim=dim_current,
                                     input_resolution=input_resolution_current,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     shift_size=0 if
                                     (i % 2 == 0) else window_size // 2,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(
                                         drop_path, list) else drop_path,
                                     norm_layer=norm_layer)
                for i in range(depth)
            ]
            if l < (layer_depth - 1) and downsample is not None:
                blocks_list.append(
                    downsample(input_resolution_current,
                               dim=dim_current,
                               norm_layer=norm_layer))
            layer_list += blocks_list

        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):

        x = self.layer(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=2,
                 embed_dim=96,
                 norm_layer=nn.LayerNorm,
                 activation=nn.GELU):
        super().__init__()
        img_size = to_2tuple(
            img_size)  # 224-->(224, 224) or [224, 448]-->[224, 448]
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

        self.activation = activation

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans,
                      embed_dim // 3,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # self.activation(),
            nn.Conv2d(embed_dim // 3,
                      embed_dim // 3,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # self.activation(),
            nn.Conv2d(embed_dim // 3,
                      embed_dim,
                      kernel_size=patch_size,
                      stride=patch_size),
            # self.activation(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x)
        x = self.layers(x)

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_scale=False):
    _, _, h, w = target_as.size()
    if if_scale:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= (w / w_)
        inputs[:, 1, :, :] *= (h / h_)
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return res
