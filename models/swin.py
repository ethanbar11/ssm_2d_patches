# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_
# from utils.drop_path import DropPath
import torch
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization
from einops import rearrange
from .mega.moving_average_gated_attention import MovingAverageGatedAttention
from .mega.mega_layer import MegaLayer
from .mega.exponential_moving_average import MultiHeadEMA
from .mega.two_d_ssm_recursive import TwoDimensionalSSM
# from src.models.sequence.modules.s4nd import S4ND
from .mix_ffn import MixFFN


def drop_path(x, dim, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[dim],) + (1,) * (x.ndim - dim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None, dim=0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.dim = dim
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.dim, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return 'p={}, dim={}, keep_scale={}'.format(self.drop_prob, self.dim, self.scale_by_keep)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
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
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 is_LSA=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        print('dim', dim, 'head_dim', head_dim, 'num_heads', num_heads)
        self.scale = qk_scale or head_dim ** -0.5
        self.is_LSA = is_LSA
        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(self.num_heads))
            self.mask = torch.eye((window_size[0] ** 2), (window_size[0] ** 2))
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
            self.inf = float('-inf')

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if not self.is_LSA:
            q = q * self.scale

        else:
            scale = self.scale
            q = torch.mul(q, scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((B_, self.num_heads, 1, 1)))

        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

            if self.is_LSA:
                attn[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
            attn = self.softmax(attn)

        else:
            if self.is_LSA:
                attn[:, :, self.mask[:, 0], self.mask[:, 1]] = self.inf
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
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

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_LSA=False, args=None, save_path=None):
        super().__init__()
        self.dim = dim
        self.external_bias = True
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
        '''
        if self.shift_size == 0:
            if issubclass(norm_layer, nn.LayerNorm):
                norm_type = 'layernorm'
            print(norm_type)
            if args.no_dropout_mega:
                drop = 0 
                attn_drop = 0

            # if True:#not self.use_megaMovingAverageGatedAttention_layer:
            #     self.attn = MovingAverageGatedAttention(embed_dim=dim, zdim=dim // 4, hdim=dim * 2, ndim=args.ndim,
            #                                             attention_activation=self.attention_activation,
            #                                             patch_amount=window_size ** 2, dropout=drop,
            #                                             attention_dropout=attn_drop, hidden_dropout=attn_drop,
            #                                             no_rel_pos_bias=True,
            #                                             drop_path=drop_path, norm_type=norm_type, args=args)
            # else:
            #     mega_layer = MegaLayer(embed_dim = dim ,hidden_dim = dim * 2, z_dim = dim // 4  ,n_dim = args.ndim,ffn_embed_dim = dim , dropout = drop, attention_dropout = attn_drop, hidden_dropout = 0.0,activation_dropout = 0.,
            #     drop_path=drop_path, chunk_size = -1, truncation=None, max_positions = 1024, activation='silu', attention_activation=self.attention_activation, norm_type ='layernorm', no_rel_pos_bias = False) 
            #     self.attn = mega_layer

        else:
        '''
        self.use_mega_gating = args.use_mega_gating
        if args.use_mega_gating:
            if issubclass(norm_layer, nn.LayerNorm):
                norm_type = 'layernorm'
            print(norm_type)
            if args.no_dropout_mega:
                drop = 0
                attn_drop = 0
            real_ema = args.ema
            args.ema = None
            zdim = math.floor(args.zdim_ratio * dim) if args.zdim_ratio is not None else dim // 4
            hdim = math.floor(args.hidden_dim_ratio * dim) if args.hidden_dim_ratio is not None else dim * 2
            self.attn = MovingAverageGatedAttention(embed_dim=dim, zdim=zdim, hdim=hdim, ndim=args.ndim,
                                                    # attention_activation='relu',
                                                    heads_num=num_heads,
                                                    patch_amount=window_size ** 2, dropout=drop,
                                                    attention_dropout=attn_drop, hidden_dropout=attn_drop,
                                                    no_rel_pos_bias=True,
                                                    drop_path=drop_path, norm_type=norm_type, args=args)
            args.ema = real_ema

        else:
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, is_LSA=is_LSA)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if args.use_mix_ffn:
            self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              num_patches=self.input_resolution[0] ** 2, args=args)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # N_w^2, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # N_w^2, window_size, window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(
                2)  # (N_w^2, 1, window_size, window_size) - (N_w^2, window_size, 1, window_size)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)  # No parameter

        if args.ema == 'ssm_2d' and not args.use_mix_ffn:
            self.move = TwoDimensionalSSM(embed_dim=dim, L=self.input_resolution[0] ** 2, args=args)

        elif args.ema == 's4nd':
            config_path = args.s4nd_config
            # Read from config path with ymal
            import yaml
            config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
            config['n_ssm'] = args.n_ssm
            config['d_state'] = args.ndim
            self.move = S4ND(**config, d_model=self.dim, l_max=int(math.sqrt(self.input_resolution[0] ** 2)),
                             return_state=False)
            print('S4ND', self.move)
        elif args.ema == 'ema':
            self.move = MultiHeadEMA(embed_dim=dim, ndim=args.ndim, bidirectional=True, truncation=None)

        else:
            self.move = nn.Identity()

        # Print the total amount of params
        tot = 0
        for name, w in dict(self.move.named_parameters()).items():
            print(name, w.shape, w.numel())
            tot += w.numel()
        print('Total params in:', args.ema, tot)

    def forward(self, x):
        # H, W = self.input_resolution
        B, L, C = x.shape
        H = int(math.sqrt(L))
        #        assert L == H * W, "input feature has wrong size"
        #        print(H, W, B, L, C)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, H, C)

        # ! EMA:
        if self.external_bias:
            sh = x.shape
            x = rearrange(x, 'b l1 l2 h -> (l1 l2) b h')
            x = self.move(x)
            x = rearrange(x, '(l1 l2) b h -> b l1 l2 h', l1=H)
            assert x.shape == sh

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.use_mega_gating:
            sh = x_windows.shape
            x_windows = rearrange(x_windows, 'b l h -> l b h')
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
            attn_windows = rearrange(attn_windows, 'l b h -> b l h')
            x_windows = rearrange(x_windows, 'l b h -> b l h')
            assert x_windows.shape == sh

        else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        '''
        if self.shift_size == 0:
            x_windows = rearrange(x_windows, 'b l h -> l b h')
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows = rearrange(attn_windows, 'l b h -> b l h')
        else:
            # Mega(no ema, only gating) or Regular
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        '''
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, H)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, L, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        #       assert L == H * W, "input feature has wrong size"
        #       assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

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
        flops = H * W * self.dim  # layer norm
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim  # reduction
        return flops


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

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=False, use_checkpoint=False,
                 is_LSA=False, is_SPT=False, args=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, is_LSA=is_LSA, args=args)
            # , save_path=args.save_path.format(i))
            for i in range(depth)])

        # patch merging layer
        if downsample:
            if not is_SPT:
                self.downsample = PatchMerging(input_resolution, dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = ShiftedPatchTokenization(dim, dim * 2, 2)

        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                # print(x.shape)
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)

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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


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
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, is_LSA=False, is_SPT=False, args=None,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        if args.save_directory is not None:
            self.save_and_exit = True
        else:
            self.save_and_exit = False

        """ Base """
        if not is_SPT:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
            self.img_resolution = self.patch_embed.patches_resolution

        else:
            self.patch_embed = ShiftedPatchTokenization(3, embed_dim, patch_size, is_pe=True)
            self.img_resolution = (img_size // patch_size, img_size // patch_size)

            # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.img_resolution[0] ** 2, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.pool_idx = list()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_first = i_layer == 0
            if args.save_directory:
                args.save_path = os.path.join(args.save_directory, 'layer_{}'.format(i_layer), 'block_{}')
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.img_resolution[0] // (2 ** i_layer),
                                  self.img_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, is_LSA=is_LSA, is_SPT=is_SPT,
                downsample=True if (i_layer < self.num_layers - 1) else False,
                use_checkpoint=use_checkpoint, args=args)
            self.layers.append(layer)

        self.img_resolution = [self.img_resolution[0] // (2 ** (self.num_layers - 1)),
                               self.img_resolution[1] // (2 ** (self.num_layers - 1))]

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        k = 0

        x = self.patch_embed(x)

        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        if self.save_and_exit:
            exit()
        return x
