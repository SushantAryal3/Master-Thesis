import sys
from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
from torch import einsum
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import numpy as np
from typing import List
from pathlib import Path
import zarr
from torch.utils.data import Dataset
from typing import Optional, Tuple
import albumentations as A
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import csv
import torch
import torchmetrics
from typing import Tuple
import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform

def get_norm3d(name,channels,num_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm3d(num_features=channels)
    elif name == 'GroupNorm':
        return torch.nn.GroupNorm(num_channels=channels,num_groups=num_groups)
    else:
        raise ValueError("I do not understand normalization name::{}, options:: BatchNorm, GroupNorm, aborting ...".format(name))

def get_norm2d(name,channels,num_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm2d(num_features=channels)
    elif name == 'GroupNorm':
        return torch.nn.GroupNorm(num_channels=channels,num_groups=num_groups)
    else:
        raise ValueError("I do not understand normalization name::{}, options:: BatchNorm, GroupNorm, aborting ...".format(name))
    
class Conv3DNormed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=(1, 1,1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), norm_type = 'BatchNorm', num_groups=None, 
                 groups=1):
        super(Conv3DNormed,self).__init__()

        self.conv3d = torch.nn.Conv3d(in_channels=in_channels, 
                                      out_channels=out_channels,
                                      kernel_size= kernel_size,
                                      stride= strides,
                                      padding=padding,
                                      dilation= dilation,
                                      bias=False,
                                      groups=groups)
        self.norm_layer = get_norm3d(name=norm_type,channels=out_channels,num_groups=num_groups)

    @torch.jit.export
    def forward(self,input:torch.Tensor):
        x = self.conv3d(input)
        x = self.norm_layer(x)
        return x

class D2SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        u = torch.special.expit(-x)
        ctx.save_for_backward(u)
        return u*(1. + u*(-3. + 2.*u))
        
    @staticmethod
    def backward(ctx,grad_output):
        u = ctx.saved_tensors[0]
        return u*(-1. + u*(7. + u*(-12. + 6.*u)))*grad_output

class D2Sigmoid(torch.nn.Module):
    def __init__(self,scale=False):
        super(D2Sigmoid,self).__init__()
    def forward(self,input):
        return D2SigmoidFunction.apply(input)
    
class Patchify3DCHW(torch.nn.Module):
    def __init__(self, cscale, hscale, wscale):
        super().__init__()
        self.c = cscale
        self.h = hscale
        self.w = wscale
        self.unfold_shape = None

    def _2patch(self, input):
        shape = input.shape
        c = torch.div(shape[-3], self.c, rounding_mode="floor")
        h = torch.div(shape[-2], self.h, rounding_mode="floor")
        w = torch.div(shape[-1], self.w, rounding_mode="floor")

        sc = c
        sh = h
        sw = w

        patch = input.unfold(2, c, sc).unfold(3, h, sh).unfold(4, w, sw)
        self.unfold_shape = patch.shape
        return patch

    def _2tensor(self, patch):
        if self.unfold_shape is None:
            raise RuntimeError("unfold_shape is None. Call _2patch() before _2tensor().")

        output_c = self.unfold_shape[2] * self.unfold_shape[5]
        output_h = self.unfold_shape[3] * self.unfold_shape[6]
        output_w = self.unfold_shape[4] * self.unfold_shape[7]

        tensorpatch = patch.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
        tensorpatch = tensorpatch.view(
            self.unfold_shape[0],
            self.unfold_shape[1],
            output_c,
            output_h,
            output_w,
        )
        return tensorpatch

    def forward(self, x):
        return self._2patch(x)


class BASE_RelPatchAttention3D_TCHW(torch.nn.Module):
    def __init__(self, nfilters, scales, correlation_method="sum", TimeDim=None, depth=0.0):
        super().__init__()

        self.alpha = 2.0 ** depth
        self.beta = 2.0 * self.alpha - 1

        if depth == 0.0:
            self.qk_sim = self._qk_identity_sim_v1
        else:
            self.qk_sim = self._qk_identity_sim_v2

        self.scales = scales
        self.patchify = Patchify3DCHW(cscale=scales[0], hscale=scales[1], wscale=scales[2])

        if correlation_method == "sum":
            self.qk_compact = self._qk_compact_v1
        elif correlation_method == "mean":
            self.qk_compact = self._qk_compact_v2
        elif correlation_method == "linear":
            if TimeDim is None:
                raise ValueError("TimeDim must be provided when correlation_method='linear'.")
            self.shrink_2_1 = torch.nn.Linear(
                in_features=TimeDim * scales[0] * scales[1] * scales[2],
                out_features=1,
            )
            self.qk_compact = self._qk_compact_v3
        else:
            raise ValueError("Cannot understand correlation method, aborting ...")

    def _qk_identity_sim_v1(self, q, k, smooth=1.0e-5):
        scale = np.reciprocal(np.sqrt(np.prod(q.shape[-3:])))
        q = q * scale
        k = k * scale

        qk = einsum("iWjklmno,iPstrmno->iWjklPstr", q, k)
        qq = einsum("iWjklmno,iWjklmno->iWjkl", q, q)
        kk = einsum("iPstrmno,iPstrmno->iPstr", k, k)

        denum = (qq[:, :, :, :, :, None, None, None, None] + kk[:, None, None, None, None]) - qk + smooth
        logqk = torch.log(qk + smooth)
        logdenum = torch.log(denum)
        result = torch.exp(logqk - logdenum)
        return result

    def _qk_identity_sim_v2(self, q, k, smooth=1.0e-5):
        scale = np.reciprocal(np.sqrt(np.prod(q.shape[-3:])))
        q = q * scale
        k = k * scale

        qk = einsum("iWjklmno,iPstrmno->iWjklPstr", q, k)
        qq = einsum("iWjklmno,iWjklmno->iWjkl", q, q)
        kk = einsum("iPstrmno,iPstrmno->iPstr", k, k)

        denum = self.alpha * (qq[:, :, :, :, :, None, None, None, None] + kk[:, None, None, None, None]) - self.beta * qk + smooth
        logqk = torch.log(qk + smooth)
        logdenum = torch.log(denum)
        result = torch.exp(logqk - logdenum)
        return result

    def _qk_compact_v1(self, qk):
        return torch.sum(qk, dim=1)

    def _qk_compact_v2(self, qk):
        return torch.mean(qk, dim=1)

    def _qk_compact_v3(self, qk):
        tqk = qk.permute(0, 2, 3, 4, 5, 1)
        tqk2 = self.shrink_2_1(tqk).squeeze(dim=-1)
        return tqk2

    def qk_select_v(self, qk, vpatch, smooth=1.0e-5):
        tqk = qk.reshape(qk.shape[0], -1, *qk.shape[5:])
        tqk = self.qk_compact(tqk)
        qkvv = einsum("bTrst, bTrstmno -> bTrstmno", tqk, vpatch)
        qkvv = self.patchify._2tensor(qkvv)
        return qkvv

    def get_att(self, q, k, v):
        qp = self.patchify._2patch(q)
        kp = self.patchify._2patch(k)
        vp = self.patchify._2patch(v)

        qpkp = self.qk_sim(qp, kp)
        vout = self.qk_select_v(qpkp, vp)
        return vout

class RelPatchAttention3DTCHW(torch.nn.Module):
    def __init__(self,in_channels,out_channels,scales,kernel_size=(3,3,3),padding=(1,1,1),nheads=1,norm='BatchNorm',norm_groups=None, correlation_method='sum',  TimeDim=None,depth=0.0):
        super().__init__()
        self.act = D2Sigmoid(scale = False)
        self.patch_attention = BASE_RelPatchAttention3D_TCHW(out_channels, scales, correlation_method=correlation_method,TimeDim=TimeDim,depth=depth)
        self.query   = Conv3DNormed(in_channels=in_channels,out_channels=out_channels,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)
        self.kv      = Conv3DNormed(in_channels=in_channels,out_channels=out_channels*2,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads*2)

    def forward(self, input1:torch.Tensor, input2: torch.Tensor):
        q = self.query(input1)
        k,v = self.kv(input2).split(q.shape[1],1)
        q = self.act(q)+0.1
        k = self.act(k)+0.1
        q = q.permute(0,2,1,3,4)
        k = k.permute(0,2,1,3,4)
        v = v.permute(0,2,1,3,4)
        v = self.patch_attention.get_att(q,k,v)
        v = v.permute(0,2,1,3,4)
        v = self.act(v)
        return v

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
    def forward(self, x):
        device = x.device
        if self.prob == 0. or (not self.training):
            return x
        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)
        
class MBConvResidual3D(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super(). __init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class SqueezeExcitation3D(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        self.gate = nn.Sequential(
            Reduce('b c s h w -> b s c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b s c -> b c s 1 1')
        )
    def forward(self, x):
        return x * self.gate(x)
    
def MBConv3D(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.,
):
    hidden_dim = int(expansion_rate * dim_out)
    # Only downsample spatially (H,W). Time dimension is not downsampled.
    stride = (1,2,2) if downsample else (1,1,1)
    conv3d_spatial = nn.Conv3d(hidden_dim, hidden_dim, (3,3,3), stride = stride, padding = (1,1,1), groups = hidden_dim)
    net = nn.Sequential(
        # Expand (pointwise 1x1x1). It's kernel size is 1 in time and space, so it mixes only across channels, not across pixels or time frames
        nn.Conv3d(dim_in, hidden_dim, 1),
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        # 3×3×3 DEPTHWISE conv to mix local spatio-temporal context
        conv3d_spatial,
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        ## Each channel now encodes a different learned pattern for example (vegetation, moisture, temporal change intensity). But the model doens't yet
        ## know which channel are more relevant for the current sample. All channel are treated equally. The SE3D block lets the model dynamically reweight
        ## channels.
        SqueezeExcitation3D(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv3d(hidden_dim, dim_out, 1),
        nn.BatchNorm3d(dim_out)
    )
    if dim_in == dim_out and not downsample:
        net = MBConvResidual3D(net, dropout = dropout)
    return net

class LayerNormChannelsLast3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        x = rearrange(input,'b c s h w-> b s h w c')
        x = self.norm(x)
        x = rearrange(x,'b s h w c -> b c s h w')
        return x

class PreNormResidualAtt3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return (self.fn(self.norm(x)) + 1)*x

class PTAttention3DTCHW(nn.Module):
    def __init__(
        self,
        dim,
        nheads = 32,
        dropout = 0.,
        scales = (4,4),
        verbose=False,
        correlation_method='mean',
        TimeDim=None,
        depth=10.0
    ):
        super().__init__()
        if verbose:
            print("nfilters::{}, scales::{}, nheads::{}".format(dim, scales,nheads))
        self.att      =  RelPatchAttention3DTCHW(
                                in_channels  	   = dim,
                                out_channels 	   = dim,
                                nheads       	   = nheads,
                                scales       	   = scales,
                                norm         	   = 'GroupNorm',
                                norm_groups  	   = dim//4,
                                correlation_method = correlation_method,
                                TimeDim            = TimeDim, 
                                depth 		   = depth)

    def forward(self,input):       
        return   self.att(input,input)

class PreNormResidual3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class FeedForward3D(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = rearrange(x,'b c s h w -> b s h w c')
        x = self.net(x)
        return rearrange(x,'b s h w c -> b c s h w')

class PTAViTStage3DTCHW(nn.Module):
    def __init__(
        self,
        layer_dim_in,
        layer_dim,
        layer_depth,
        nheads,
        scales,
        downsample=False,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        correlation_method="mean",
        TimeDim = None,
        depth = 10.0
    ):
        super(). __init__()
        stage = []
        for stage_ind in range(layer_depth):
            is_first = stage_ind == 0
            block = nn.Sequential(
                MBConv3D(
                    layer_dim_in if is_first else layer_dim,
                    layer_dim,
                    downsample = downsample if is_first else False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate,
                ),
                PreNormResidualAtt3D(
                    layer_dim,
                    PTAttention3DTCHW(
                        dim = layer_dim,
                        nheads = nheads,
                        dropout = dropout,
                        scales = scales,
                        correlation_method = correlation_method,
                        TimeDim = TimeDim,
                        depth = depth
                    )
                ),
                PreNormResidual3D(
                    layer_dim,
                    FeedForward3D(
                        dim = layer_dim,
                        dropout = dropout
                    )
                )
            )
            stage.append(block)

        self.stage = torch.nn.Sequential(*stage)

    def forward(self, input):
        return self.stage(input)

class UpSample2D3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest', norm_type = 'BatchNorm', norm_groups=None,causal=False):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=(1,scale_factor,scale_factor), mode=mode)
        self.convup_normed= Conv3DNormed(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3,3,3),
                    padding=(1,1,1),
                    norm_type=norm_type,
                    num_groups=norm_groups)
    def forward(self,input):
        out = self.upsample(input)
        out = self.convup_normed(out)
        return out
    
class combine_layers3D(torch.nn.Module):
    def __init__(self,nfilters,  norm_type = 'BatchNorm', norm_groups=None,causal=False):
        super().__init__()
        self.up = UpSample2D3D(in_channels=nfilters*2, out_channels=nfilters, norm_type = norm_type, norm_groups=norm_groups, causal=causal)
        self.conv_normed = Conv3DNormed(in_channels = 2*nfilters,out_channels=nfilters,
                                            kernel_size=(1,1,1),
                                            padding=(0,0,0),
                                            norm_type=norm_type,
                                            num_groups=norm_groups)

    def forward(self,_layer_lo, _layer_hi):
        up = self.up(_layer_lo)
        up = torch.relu(up)
        x = torch.cat([up,_layer_hi], dim=1)
        x = self.conv_normed(x)
        return x

class FuseHiLo(torch.nn.Module):
    def __init__(self, nfilters, nfilters_embed=96, scales=(4,8,8),   norm_type = 'BatchNorm', norm_groups=None,depth=10.0):
        super().__init__()
        self.embedding1 = Conv3DNormed(in_channels = nfilters, out_channels = nfilters_embed, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)
        self.embedding2 = Conv3DNormed(in_channels = nfilters, out_channels = nfilters_embed, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)
        self.upscale = UpSample2D3D(in_channels=nfilters_embed,out_channels=nfilters_embed,scale_factor=4,norm_type=norm_type,norm_groups=norm_groups)
        self.conv3d = Conv3DNormed(in_channels=nfilters_embed*2, out_channels = nfilters_embed,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.att = RelPatchAttention3DTCHW(in_channels=nfilters_embed, out_channels = nfilters_embed,nheads=nfilters_embed//4,norm=norm_type,norm_groups=norm_groups,
                scales=scales,
                depth=depth
              )
    def forward(self, UpConv4, conv1):
        UpConv4 = self.embedding1(UpConv4)
        UpConv4 = self.upscale(UpConv4)
        conv1   = self.embedding2(conv1)
        convl = torch.cat([conv1,UpConv4],dim=1)
        conv = self.conv3d(convl)
        conv = torch.relu(conv)
        conv = conv * (1.+self.att(conv,conv))
        return conv

class ptavit3d_dn_features(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nfilters_embed=32, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4, correlation_method='mean', TimeDim=None, attention_depth=0.0,stem_norm=True):
        super().__init__()
        
        def closest_power_of_2(num_array):
            # num_array = [5,27,58,69]
            log2_array = np.log2(num_array)
            # [2.32, 4,75, 5.85, 6.10]
            rounded_log2_array = np.round(log2_array)
            # [2,5,6,6]
            closest_power_of_2_array = np.power(2, rounded_log2_array)
            # [4,32,64,64]
            return np.maximum(closest_power_of_2_array, 1).astype(int)

        def resize_scales(channel_size, spatial_size, scales_all):
            temp = np.array(scales_all)*np.array([channel_size/96,spatial_size[0]/256,spatial_size[1]/256])
            return closest_power_of_2(temp).tolist()

        scales_all = [[16,16,16],[32,8,8],[64,4,4],[128,2,2],[128,2,2],[128,1,1],[256,1,1],[256,1,1]]
        scales_all = resize_scales(nfilters_init, spatial_size_init,scales_all)
        # 128 x 128 [[16,8,8],[32,4,4],[64,2,2],[128,1,1],[128,1,1],[128,1,1],[256,1,1],[256,1,1]]
        # 256 x 256 [[16,16,16],[32,8,8],[64,4,4],[128,2,2],[128,2,2],[128,1,1],[256,1,1],[256,1,1]]
        self.scales_all = scales_all

        self.depth = depth = len(depths)
        num_stages = len(depths)
        # num_stages = 4
        dims = tuple(map(lambda i: (2 ** i) * nfilters_init, range(num_stages)))
        dims = (nfilters_init, *dims)
        # dims = (96,96,192,384,768)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        # dims_pair = ((96,96),(96,192),(192,384),(384,768))

        self.conv1 = Conv3DNormed(in_channels=in_channels, out_channels = nfilters_init, kernel_size=1,padding=0,strides=1, norm_type=norm_type, num_groups=norm_groups)

        if stem_norm: # use the when the batch size is 1 or 2. 
            self.conv_stem = nn.Sequential(
            ## reduce the hight and width by half. (so output (B,nfilter_init,T,H/2,W/2))
            nn.Conv3d( nfilters_init, nfilters_init, (3,3,3), stride = (1,2,2), padding = (1,1,1)),
            nn.GroupNorm(num_groups=norm_groups, num_channels=nfilters_init),
            nn.Conv3d(nfilters_init, nfilters_init, 3, padding = 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=nfilters_init)
            )
        else:
            self.conv_stem = nn.Sequential(
            nn.Conv3d( nfilters_init, nfilters_init, (3,3,3), stride = (1,2,2), padding = (1,1,1)),
            nn.Conv3d(nfilters_init, nfilters_init, 3, padding = 1)
            )
        # if spatial_size_init = (128,128) then it becomes (32,32)
        spatial_size_init = tuple(ts // 4 for ts in spatial_size_init)
        self.stages_dn = [] # Store the encoder stages
        if verbose:
            print (" @@@@@@@@@@@@@ Going DOWN @@@@@@@@@@@@@@@@@@@ ")
        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            # index (1,2,3,4) = value(24,48,96,192)
            nheads = nheads_start * 2**idx # used from group in conv3d block
            scales = scales_all[idx]
            # if spatial_size_init = 128 then spatial_size 1:(32,32), 2:(16,16), 3:(8,8), 4:(4,4)
            spatial_size = tuple( ts // 2**idx for ts in spatial_size_init )
            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(idx,layer_dim_in,layer_dim,layer_depth,spatial_size, scales)) 
            self.stages_dn.append(PTAViTStage3DTCHW(
                layer_dim_in=layer_dim_in,
                layer_dim=layer_dim,
                layer_depth=layer_depth,
                nheads=nheads,
                scales=scales,
                downsample=True,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,
                correlation_method=correlation_method,
                TimeDim=TimeDim,
                depth=attention_depth
                ))
        self.stages_dn = torch.nn.ModuleList(self.stages_dn)
        self.stages_up = [] 
        self.UpCombs  = [] 
        dim_pairs = dim_pairs[::-1] # [(384, 768), (192, 384), (96, 192), (96, 96)]
        depths    = depths[::-1]    # [2, 5, 2, 2]
        dim_pairs = dim_pairs[:-1]  # [(384, 768), (192, 384), (96, 192)]
        depths    = depths[1:]      # [5, 2, 2]
        if verbose:
            print (" XXXXXXXXXXXXXXXXXXXXX Coming up XXXXXXXXXXXXXXXXXXXXXXXXX " )
        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
            idx = len(depths)-1 - idx   # (2,1,0)
            nheads = int(nheads_start * 2**(idx)) # (96,48,24)
            spatial_size = tuple( ts // 2**idx for ts in spatial_size_init ) #((8,8),(16,16),(32,32))
            scales = scales_all[idx] # ([64,2,2],[32,8,8],[16,16,16])
            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(2*depth-idx-2, 
                    layer_dim_in, layer_dim_in, layer_depth,spatial_size, scales))
            self.stages_up.append(PTAViTStage3DTCHW(
                layer_dim_in	      = layer_dim_in,
                layer_dim	      = layer_dim_in,
                layer_depth	      = layer_depth,
                nheads		      = nheads,
                scales		      = scales,
                downsample	      = False,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout 	      =	0.1,
                correlation_method    =	correlation_method,
                TimeDim		      =	TimeDim,
                depth                 = attention_depth
                ))
            self.UpCombs.append(combine_layers3D(
                layer_dim_in, 
                norm_type=norm_type,
                norm_groups=norm_groups))
        self.stages_up   = torch.nn.ModuleList(self.stages_up)
        self.UpCombs    = torch.nn.ModuleList(self.UpCombs)
        self.fuse_hi_lo = FuseHiLo( nfilters=layer_dim_in, nfilters_embed=nfilters_embed, scales=(4,8,8),   norm_type = norm_type, norm_groups=norm_groups,depth=attention_depth)

    def forward(self, input_t1):
        conv1_t1 = self.conv1(input_t1)
        conv1 = self.conv_stem(conv1_t1)
        fusions = []
        for idx in range(self.depth):
            conv1 = self.stages_dn[idx](conv1)
            fusions.append(conv1)
        convs_up = fusions[-1]
        convs_up = torch.relu(convs_up)
        for idx in range(self.depth-1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            convs_up = self.stages_up[idx](convs_up)
        final = self.fuse_hi_lo(convs_up, conv1_t1)
        return final

class Conv2DNormed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=(1, 1),
                 padding=(0, 0), dilation=(1, 1), norm_type = 'BatchNorm', num_groups=None, 
                 groups=1):
        super(Conv2DNormed,self).__init__()

        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels,
                                      kernel_size= kernel_size,
                                      stride= strides,
                                      padding=padding,
                                      dilation= dilation,
                                      bias=False,
                                      groups=groups)
        self.norm_layer = get_norm2d(name=norm_type,channels=out_channels,num_groups=num_groups)

    @torch.jit.export
    def forward(self,input:torch.Tensor):
        x = self.conv2d(input)
        x = self.norm_layer(x)
        return x

class SigmoidCrisp(torch.nn.Module):
    def __init__(self,smooth=1.e-2):
        super(SigmoidCrisp,self).__init__()
        self.smooth = smooth
        self.gamma = torch.nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self,input):
        out = self.smooth + torch.sigmoid(self.gamma)
        out = torch.reciprocal(out)
        out = input*out 
        out = torch.sigmoid(out)
        return out 

class HeadSingle(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  NClasses, depth=2, norm_type='BatchNorm',norm_groups=None, **kwargs):
        super().__init__(**kwargs)
        logits = [] 
        logits.append( Conv2DNormed(in_channels = in_channels, out_channels = out_channels, kernel_size = (3,3),padding=(1,1), norm_type=norm_type, num_groups=norm_groups))
        for _ in range(depth-1):
            logits.append( Conv2DNormed(in_channels = out_channels, out_channels = out_channels,kernel_size = (3,3),padding=(1,1), norm_type=norm_type, num_groups=norm_groups))
            logits.append( torch.nn.ReLU())
        logits.append( torch.nn.Conv2d(in_channels=out_channels, out_channels=NClasses,kernel_size=1,padding=0))
        self.logits = torch.nn.Sequential(*logits)

    def forward(self,input):
        return self.logits(input)

class head_cmtsk(torch.nn.Module):
    def __init__(self, nfilters, NClasses, nfilters_embed=32, spatial_size=256,scales=(4,8),   norm_type = 'BatchNorm', norm_groups=None,segm_act ='softmax'):
        super().__init__()
        self.model_name = "Head_CMTSK_BC" 
        self.nfilters = nfilters_embed
        self.NClasses = NClasses
        self.distance_logits = HeadSingle(in_channels = nfilters_embed, out_channels = nfilters_embed,  NClasses = NClasses, norm_type = norm_type, norm_groups=norm_groups)
        self.dist_Equalizer = Conv2DNormed(in_channels = NClasses, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.Comb_bound_dist =  Conv2DNormed(in_channels= nfilters_embed*2, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.bound_logits = HeadSingle(in_channels = nfilters_embed*2, out_channels = nfilters_embed,NClasses=NClasses, norm_type = norm_type, norm_groups=norm_groups)
        self.bound_Equalizer = Conv2DNormed(in_channels=NClasses, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.final_segm_logits = HeadSingle(in_channels = nfilters_embed*2, out_channels=nfilters_embed, NClasses = NClasses, norm_type = norm_type, norm_groups=norm_groups)
        self.CrispSigm = SigmoidCrisp()
        if ( self.NClasses == 1):
            self.ChannelAct = SigmoidCrisp()
            self.segm_act   = SigmoidCrisp() 
        else:
            if segm_act =='softmax':
                self.segm_act   = torch.nn.Softmax(dim=1)  
            elif segm_act =='sigmoid':
                self.segm_act   = SigmoidCrisp() 
            else:
                raise ValueError("I don't understand type of segm_act, aborting ...")
            self.ChannelAct = torch.nn.Softmax(dim=1) 

    def forward(self, conv):
        dist = self.distance_logits(conv)
        dist = self.ChannelAct(dist)
        distEq = torch.relu(self.dist_Equalizer(dist))
        bound = torch.cat([conv, distEq],dim=1)
        bound = self.bound_logits(bound)
        bound   = self.CrispSigm(bound) 
        boundEq = torch.relu(self.bound_Equalizer(bound))
        comb_bd = self.Comb_bound_dist(torch.cat([boundEq, distEq],dim=1))
        comb_bd = torch.relu(comb_bd)
        all_layers = torch.cat([comb_bd, conv],dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.segm_act(final_segm)
        return  torch.cat([final_segm, bound, dist],dim=1)

__all__=['FTanimoto']

def inner_prod(prob, label,axis:List[int]):
    return (prob * label).sum(dim=axis,keepdim=True)

class tnmt_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p:torch.Tensor,l:torch.Tensor,d:int,axis:List[int]=[2,3]): 
        pl = inner_prod(p,l,axis)
        pp = inner_prod(p,p,axis)
        ll = inner_prod(l,l,axis)
        a = 2**d
        b = -(2.*a-1.)
        denum = a*(pp+ll) + b*pl
        scale = torch.reciprocal(denum)
        scale = torch.nan_to_num(scale, nan=0.0,posinf=1.,neginf=-1)
        ctx.save_for_backward(p,l,pl,pp,ll,scale)
        ctx.a = a
        result = pl*scale
        return result
    
    @staticmethod 
    def backward(ctx,grad_output):
        p, l, pl, pp, ll, scale = ctx.saved_tensors
        a = ctx.a
        ascale2 = (a*scale)*scale
        ppmll  = pp+ll
        result_p = ascale2 *(-2.*p*pl + l *ppmll)
        result_l = ascale2 *(-2.*l*pl + p *ppmll)
        return result_p  * grad_output, result_l  * grad_output, None, None     


class FTanimoto(torch.nn.Module):
    def __init__(self, depth=0, axis=[2,3],mode='exact'):
        super().__init__()
        if depth == 0:
            self.scale=1.
        else:
            self.scale = 1./(depth+1.)
        self.depth=depth
        self.axis=axis
        if mode=='exact' or depth==0:
            self.tnmt_base = self.tnmt_base_exact
        elif mode=='avg':
            self.tnmt_base = self.tnmt_base_avg
        else:
            raise  ValueError("variable mode must be one of 'avg' or 'exact', default == 'avg'")

    def set_depth(self,depth):
        assert depth >= 0, "Expecting depth >= 0, aborting ..."
        if depth == 0:
            scale=1.
        else:
            scale = 1./(depth+1.)
        self.scale = torch.tensor(scale)
        self.depth = depth 

    @torch.jit.export
    def tnmt_base_avg(self, preds, labels):
        if self.depth==0:
            return tnmt_2d.apply(preds,labels,self.depth,self.axis)
        else:
            result = 0.0
            for d in range(self.depth+1):
                result = result + tnmt_2d.apply(preds,labels,d,self.axis)
            return result * self.scale

    @torch.jit.export
    def tnmt_base_exact(self, preds, labels):
        return tnmt_2d.apply(preds,labels,self.depth,self.axis)

    def forward(self, preds, labels):
            l12 = self.tnmt_base(preds,labels)
            l12 = l12 + self.tnmt_base(1.-preds, 1.-labels)
            return 0.5*l12

class ftnmt_loss(torch.nn.Module):
    def __init__(self, depth=0, axis=[2,3], mode='exact'):
        super(ftnmt_loss,self).__init__()
        self.ftnmt = FTanimoto(depth=depth, axis=axis,mode=mode)
        
    def forward(self,preds,labels):
        sim = self.ftnmt(preds,labels)
        return (1. - sim).mean()

def mtsk_loss(preds, labels,criterion, NClasses):
    pred_segm  = preds[:,:NClasses]
    pred_bound = preds[:,NClasses:2*NClasses]
    pred_dists = preds[:,2*NClasses:3*NClasses]
    label_segm  = labels[:,:NClasses]
    label_bound = labels[:,NClasses:2*NClasses]
    label_dists = labels[:,2*NClasses:3*NClasses]
    loss_segm  = criterion(pred_segm,   label_segm)
    loss_bound = criterion(pred_bound, label_bound)
    loss_dists = criterion(pred_dists, label_dists)
    return (loss_segm+loss_bound+loss_dists)/3.0

class Lambda(nn.Module):
    def __init__(self,  fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) 

class ptavit3d_dn(torch.nn.Module):
    def __init__(self, in_channels, NClasses, nfilters_init=96, nfilters_embed=96, nheads_start=96//4, depths=[2,2,5,2], spatial_size_init=(128,128), verbose=True, norm_type='GroupNorm', norm_groups=4,  correlation_method='mean',nassociations=None,segm_act='sigmoid',TimeDim=4,nblocks3d=1,attention_depth=0.0):
        super().__init__()         
        self.features = ptavit3d_dn_features(in_channels = in_channels,  spatial_size_init=spatial_size_init, nfilters_init=nfilters_init, nfilters_embed=nfilters_embed, nheads_start = nheads_start, depths = depths, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups, correlation_method=correlation_method,TimeDim=TimeDim,attention_depth=attention_depth)
        scales = self.features.scales_all[0]
        nblocks3d=nblocks3d
        self.head3D = torch.nn.Sequential(
            PTAViTStage3DTCHW(
                layer_dim_in=nfilters_embed,                
                layer_dim=nfilters_embed,                   
                layer_depth=nblocks3d,                 
                nheads=nfilters_embed//4,         
                scales=scales,                      
                downsample=False,            
                mbconv_expansion_rate = 4,   
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,               
                correlation_method='mean',   
                TimeDim=TimeDim,                
                depth=attention_depth),
            	Lambda(lambda x: x.mean(dim=2)),
            	head_cmtsk(nfilters=nfilters_init, nfilters_embed=nfilters_embed,NClasses=NClasses,
                	norm_type=norm_type,norm_groups=norm_groups)
        )

    def forward(self,input_t1):
        features3D = self.features(input_t1)
        b,c,t,h,w = features3D.shape
        preds2D3D  = self.head3D(features3D)
        return  preds2D3D 

class Classification(torchmetrics.Metric):
    def __init__(self, num_classes:int = 2, average="macro", conf_mat_multilabel=False,evaluate_conf_matrix=True, task='binary',verbose=False):
        """
        macro computes the metrics separately for each class, then average equally
        """
        super().__init__()
        self.evaluate_conf_matrix = evaluate_conf_matrix
        self.metric_acc         = torchmetrics.Accuracy(task=task,num_classes=num_classes,average=average)
        self.metric_mcc         = torchmetrics.MatthewsCorrCoef(task=task,num_classes=num_classes)
        self.metric_kappa       = torchmetrics.CohenKappa(task=task,num_classes=num_classes)
        self.metric_prec        = torchmetrics.Precision(task=task,num_classes=num_classes,average=average)
        self.metric_recall      = torchmetrics.Recall(task=task,num_classes=num_classes,average=average)
        self.metric_iou         = torchmetrics.JaccardIndex(task=task, num_classes=num_classes)
        # I added
        self.metric_f1          = torchmetrics.F1Score(task=task, num_classes=num_classes, average=average)
        #
        if self.evaluate_conf_matrix == True:
            self.metric_conf_mat    =  torchmetrics.ConfusionMatrix(task=task,num_classes=num_classes, normalize="none")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = preds.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()
        self.metric_acc(preds,target)
        self.metric_mcc(preds,target) 
        self.metric_kappa(preds,target)
        self.metric_prec(preds,target)
        self.metric_recall(preds,target)
        self.metric_f1(preds, target)
        self.metric_iou(preds, target)
        if self.evaluate_conf_matrix:
            self.metric_conf_mat(preds, target)

    def compute(self):
        out = {
            "acc"      : self.metric_acc.compute(),
            "mcc"      : self.metric_mcc.compute(),
            "kappa"    : self.metric_kappa.compute(),
            "precision": self.metric_prec.compute(),
            "recall"   : self.metric_recall.compute(),
            "iou"      : self.metric_iou.compute(),
            "f1"       : self.metric_f1.compute()
        }
        if self.evaluate_conf_matrix:
            out["conf_mat"] = self.metric_conf_mat.compute()
        return out

## Changed
class TrainingTransform(object):
    def __init__(self, prob = 1., mode='train', dist_compress=1.0):
        self.distance_scale = 1.0 / dist_compress
        self.geom_trans = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.ElasticTransform(p=1), 
                        A.GridDistortion(distort_limit=0.4,p=1.),
                        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), 
                        ],p=1.)
                    ],
            p = prob)

        if mode=='train':
            self._apply = self._transform_train
        elif mode =='valid':
            self._apply = self._transform_valid
        else:
            raise ValueError(f"mode must be 'train' or 'valid', got: '{mode}'")
    
    def _transform_valid(
        self,
        x: np.ndarray,
        y: np.ndarray,  
    ) -> Tuple[np.ndarray, np.ndarray]:
        return x, y

    def _transform_train(self, x:np.ndarray, y:np.ndarray):
        T, C, H, W = x.shape
        x_flat = x.reshape(T * C, H, W).transpose(1, 2, 0).astype(np.float32)
        y_t = y.transpose(1, 2, 0).astype(np.float32)
        result = self.geom_trans(image=x_flat, mask=y_t)
        x_out = result["image"].transpose(2, 0, 1).reshape(T, C, H, W)
        y_out = result["mask"].transpose(2, 0, 1)
        return x_out, y_out
    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._apply(x, y)

def resolve_zarr_paths(zarr_root: str, years: List[int]) -> List[Path]:
    root = Path(zarr_root)
    paths = []
    for year in years:
        candidate = root / f"sar_{year}.zarr"
        if not candidate.exists():
            raise FileNotFoundError(f"Not found: {candidate}")
        paths.append(candidate)
        print(f"[zarr] found {candidate}")
    return paths

def _split_one_zarr(zarr_path, block_size=512, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    root = zarr.open_group(str(zarr_path), mode="r")
    row_off = root["row_off"][:]
    col_off = root["col_off"][:]
    block_row = row_off // block_size
    block_col = col_off // block_size
    blocks = np.unique(np.stack([block_row, block_col], axis=1), axis=0)
    blocks = blocks[rng.permutation(len(blocks))]
    n_val  = max(1, int(round(len(blocks) * val_frac)))
    val_blocks = set(map(tuple, blocks[:n_val].tolist()))
    train_idx, val_idx = [], []
    for i in range(len(row_off)):
        b = (int(block_row[i]), int(block_col[i]))
        if b in val_blocks: val_idx.append(i)
        else:               train_idx.append(i)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)

def make_multi_splits(zarr_paths, block_size=512, val_frac=0.15, seed=42):  # remove test_frac
    train_pairs, val_pairs = [], []
    for file_idx, zp in enumerate(zarr_paths):
        tr, va = _split_one_zarr(zp, block_size, val_frac, seed=seed+file_idx)
        print(f"  [{zp.name}] train={len(tr):,}  val={len(va):,}")
        train_pairs.append((file_idx, tr))
        val_pairs.append((file_idx, va))
    return train_pairs, val_pairs

def save_multi_splits(save_dir, zarr_paths, train_pairs, val_pairs):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:  # no test
        for file_idx, local_idx in pairs:
            np.save(save_dir / f"{split_name}_idx_file{file_idx}.npy", local_idx)

def load_multi_splits(save_dir, zarr_paths):
    train_pairs, val_pairs = [], []
    for file_idx in range(len(zarr_paths)):
        for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:  # no test
            pairs.append((file_idx, np.load(save_dir / f"{split_name}_idx_file{file_idx}.npy")))
    return train_pairs, val_pairs

class MultiZarrChipDataset(Dataset):
    def __init__(self, zarr_paths, split_pairs, transform=None):
        self.zarr_paths = [Path(p) for p in zarr_paths]
        self.transform  = transform
        self._index_table = [
            (file_idx, int(local_idx))
            for file_idx, local_indices in split_pairs
            for local_idx in local_indices
        ]
        self._roots = [None] * len(zarr_paths)

    def _get_root(self, file_idx):
        if self._roots[file_idx] is None:
            self._roots[file_idx] = zarr.open_group(str(self.zarr_paths[file_idx]), mode="r")
        return self._roots[file_idx]

    def __len__(self):
        return len(self._index_table)

    def __getitem__(self, item):
        file_idx, local_idx = self._index_table[item]
        root = self._get_root(file_idx)
        x = np.clip(np.array(root["X"][local_idx], dtype=np.float32), 0.0, 1.0)
        y = np.array(root["Y"][local_idx], dtype=np.float32)[:3]
        if self.transform is not None:
            x, y = self.transform(x, y)
        meta = {
            "sample_idx": local_idx,
            "file_idx"  : file_idx,
            "zarr_name" : self.zarr_paths[file_idx].name,
            "row_off"   : int(root["row_off"][local_idx]),
            "col_off"   : int(root["col_off"][local_idx]),
            "block_id"  : int(root["block_id"][local_idx]),
            "x0"        : float(root["x0"][local_idx]),
            "y0"        : float(root["y0"][local_idx]),
        }
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), meta

    @property
    def n_time(self)    -> int: return self._get_root(0)["X"].shape[1]
    @property
    def n_bands(self)   -> int: return self._get_root(0)["X"].shape[2]
    @property
    def chip_size(self) -> int: return self._get_root(0)["X"].shape[3]
    @property
    def n_labels(self)  -> int: return 3
@torch.no_grad()
def monitor_epoch(model,epoch,val_loader,NClasses,criterion,DEBUG=False):
    model.eval()
    metric_target   = Classification(num_classes=2, task="binary").cuda()
    metric_boundary = Classification(num_classes=2, task="binary").cuda()

    tot_loss = 0.0   # sum of all overall loss per batch
    tot_segm = 0.0   # sum of the segmentation/extent loss over batch
    tot_bound = 0.0  # sum of the boundary loss over batches
    tot_dist = 0.0   # sum of the distance loss over batches
    n_batches = 0    # how many batches we've processed

    for idx, (x,y,meta) in enumerate(val_loader):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        with torch.inference_mode(), autocast():
            preds = model(x)
        ## chatgpt recommendation: added .float()
        pred_segm  = preds[:,  :NClasses].float()              # segmentation / extent logits
        pred_bound = preds[:, NClasses:2*NClasses].float()     # boundary logits
        pred_dist  = preds[:, 2*NClasses:3*NClasses].float()   # dist regression output (or logits)
        label_segm  = y[:,  :NClasses].float()
        label_bound = y[:, NClasses:2*NClasses].float()
        label_dist  = y[:, 2*NClasses:3*NClasses].float()
        ls = criterion(pred_segm,  label_segm)
        lb = criterion(pred_bound, label_bound)
        ld = criterion(pred_dist,  label_dist)
        # chatgpt
        if not (torch.isfinite(ls) and torch.isfinite(lb) and torch.isfinite(ld)):
            print(f"[VAL] non-finite loss at batch {idx} — skipping")
            print(f"  segm  loss={ls.item():.6f}  pred min/max={pred_segm.min().item():.4f}/{pred_segm.max().item():.4f}")
            print(f"  bound loss={lb.item():.6f}  pred min/max={pred_bound.min().item():.4f}/{pred_bound.max().item():.4f}")
            print(f"  dist  loss={ld.item():.6f}  pred min/max={pred_dist.min().item():.4f}/{pred_dist.max().item():.4f}")
            continue
        #
        tot_loss  += ((ls + lb + ld) / 3.0).item()
        tot_segm  += ls.item()
        tot_bound += lb.item()
        tot_dist  += ld.item()
        n_batches += 1
        metric_target.update(pred_segm, label_segm)
        metric_boundary.update(pred_bound, label_bound)
    metric_kwargs_target = metric_target.compute()
    metric_kwargs_boundary = metric_boundary.compute()
    avg_loss  = tot_loss  / max(n_batches, 1)
    avg_segm  = tot_segm  / max(n_batches, 1)
    avg_bound = tot_bound / max(n_batches, 1)
    avg_dist  = tot_dist  / max(n_batches, 1)
    print(
        f"[val] epoch {epoch:03d} | "
        f"loss={avg_loss:.4f} | "
        f"segm={avg_segm:.4f} | "
        f"bound={avg_bound:.4f} | "
        f"dist={avg_dist:.4f}"
    )
    kwargs = {
        "epoch":          epoch,
        "val_loss":       avg_loss,
        "val_loss_segm":  avg_segm,
        "val_loss_bound": avg_bound,
        "val_loss_dist":  avg_dist,
    }
    for k, v in metric_kwargs_target.items():
        kwargs[k + "_target_vV"] = v.cpu().numpy() if hasattr(v, "cpu") else v
    for k, v in metric_kwargs_boundary.items():
        kwargs[k + "_bound_vV"] = v.cpu().numpy() if hasattr(v, "cpu") else v
    return kwargs

def save_checkpoint(path, epoch, model, optimizer, scaler, best_metrics, patience_counter, best_epoch):
    tmp_path = str(path) + ".tmp"
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optim_state":      optimizer.state_dict(),
        "scaler_state":     scaler.state_dict(),
        "best_metrics":     best_metrics,
        "patience_counter": patience_counter,
        "best_epoch":       best_epoch,
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f"  [ckpt] saved checkpoint -> {path}")

def load_checkpoint(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"], ckpt["best_metrics"], ckpt["patience_counter"], ckpt["best_epoch"]

def build_scheduler(optimizer, T_0=50, T_mult=2, eta_min=1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )

def train(args):
    num_epochs = args.epochs
    batch_size = args.batch_size
    reuse_splits = args.reuse_splits
    split_dir      = Path(args.split_dir)
    zarr_paths = resolve_zarr_paths(args.zarr_root, args.years)
    CKPT_DIR       = Path(args.ckpt_dir)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    resume_path    = CKPT_DIR / "latest_checkpoint.pth"
    torch.manual_seed(0)
    NClasses = 1
    nf = 64
    verbose = True
    
    block_size = args.block_size
    val_frac = args.val_frac
    seed = args.seed
    patience = getattr(args, "patience", 10) 
    model_config = {
        "in_channels"      : 2,
        "spatial_size_init": (128, 128),
        "depths"           : [2, 2, 5, 2],
        "nfilters_init"    : nf,
        "nfilters_embed"   : nf,
        "nheads_start"     : nf // 4,
        "NClasses"         : NClasses,
        "verbose"          : verbose,
        "segm_act"         : "sigmoid",
        "TimeDim"          : 10,
        "nfilters_embed"   : nf,
    }
    
    model = ptavit3d_dn(**model_config).cuda()
    criterion = ftnmt_loss().cuda()
    optimizer = torch.optim.RAdam(model.parameters(),lr=1e-3,eps=1.e-6)
    scaler = GradScaler()
    scheduler = build_scheduler(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    # ---- Resume from checkpoint if it exists ----
    start_epoch       = 0
    patience_counter  = 0
    best_epoch        = 0
    best_metrics = {
        "val_loss": float("inf"),
        "iou":      0.0,            
        "mcc":      0.0,           
        "recall":   0.0,   
        "f1":       0.0,         
    }
 
    if resume_path.exists():
        start_epoch, best_metrics, patience_counter, best_epoch = load_checkpoint(
            resume_path, model, optimizer, scaler
        )
        start_epoch += 1
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  [resume] continuing from epoch {start_epoch}, "
              f"best_epoch={best_epoch}, patience={patience_counter}/{patience}")
    else:
        print("[info] No checkpoint found, starting fresh training.")
    
    # --- splits ---
    files_exist = all(
        (split_dir / f"{s}_idx_file{fi}.npy").exists()
        for s in ("train", "val")          # ← remove "test"
        for fi in range(len(zarr_paths))
    )
    if reuse_splits and files_exist:
        train_pairs, val_pairs = load_multi_splits(split_dir, zarr_paths)

    else:
        train_pairs, val_pairs = make_multi_splits(zarr_paths, block_size=block_size, val_frac=val_frac, seed=seed)
        save_multi_splits(split_dir, zarr_paths, train_pairs, val_pairs)

    # --- datasets & dataloaders ---
    # transform_train = TrainingTransform(mode="train")
    # transform_valid = TrainingTransform(mode="valid")

    train_ds = MultiZarrChipDataset(zarr_paths, train_pairs, transform=None)
    val_ds   = MultiZarrChipDataset(zarr_paths, val_pairs,   transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,             # ← True for training
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"\n[train] {len(train_ds):,} train chips  |  {len(val_ds):,} val chips")
    print(f"[train] batch_size={batch_size}  epochs={num_epochs}  device=cuda\n")

    log_path = CKPT_DIR / "training_log.csv"
    log_fields = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss", "val_loss_segm", "val_loss_bound", "val_loss_dist",
        # Segmentation metrics
        "segm_acc", "segm_mcc", "segm_kappa", "segm_precision",
        "segm_recall", "segm_iou", "segm_f1",
        # Boundary metrics
        "bound_acc", "bound_mcc", "bound_kappa", "bound_precision",
        "bound_recall", "bound_iou", "bound_f1",  
    ]
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)
    print(f"[log] CSV log -> {log_path}\n")

    best_model_state = None
    start = datetime.now()
    METRIC_KEYS = ("acc", "mcc", "kappa", "precision", "recall", "iou", "f1")
    def extract_metrics(kwargs, suffix):
        return {k: float(kwargs[f"{k}_{suffix}"]) for k in METRIC_KEYS}
    epoch_pbar = tqdm(range(start_epoch,num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        tot_loss = 0.0
        n_train_batches = 0

        for i, (x,y,_) in enumerate(train_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                preds = model(x)
                loss = mtsk_loss(preds.float(),y.float(),criterion, NClasses)
            #chatgpt
            if not torch.isfinite(loss):
                print(f"\n[warning] non-finite train loss at epoch {epoch+1}, batch {i}: {loss.item()}")
                continue
            #
            scaler.scale(loss).backward()
            # Calude
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #
            scaler.step(optimizer)
            scaler.update()
            tot_loss += loss.item()
            n_train_batches += 1
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        #--- Validation
        kwargs = monitor_epoch(model, epoch, valid_loader, NClasses, criterion)
        avg_train_loss = tot_loss / max(n_train_batches, 1)
        kwargs["avg_train_loss"] = avg_train_loss
        segm_metrics  = extract_metrics(kwargs, "target_vV")
        bound_metrics = extract_metrics(kwargs, "bound_vV")
        current = {
            "val_loss": kwargs["val_loss"],
            "iou":      segm_metrics["iou"],
            "mcc":      segm_metrics["mcc"],
            "recall":   segm_metrics["recall"],
            "f1":       segm_metrics["f1"],
        }
        print(
            f"epoch {epoch:03d} | "
            f"train {avg_train_loss:.4f} | "
            f"val {kwargs['val_loss']:.4f} | "
            f"segm_loss {kwargs['val_loss_segm']:.4f} | "
            f"bound_loss {kwargs['val_loss_bound']:.4f} | "
            f"dist_loss {kwargs['val_loss_dist']:.4f}\n"
            f"  segm  → iou {segm_metrics['iou']:.4f} | "
            f"mcc {segm_metrics['mcc']:.4f} | "
            f"recall {segm_metrics['recall']:.4f} | "
            f"f1 {segm_metrics['f1']:.4f} | "
            f"prec {segm_metrics['precision']:.4f}\n"
            f"  bound → iou {bound_metrics['iou']:.4f} | "
            f"mcc {bound_metrics['mcc']:.4f} | "
            f"recall {bound_metrics['recall']:.4f} | "
            f"f1 {bound_metrics['f1']:.4f} | "
            f"prec {bound_metrics['precision']:.4f}",
            flush=True 
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, current_lr, avg_train_loss,
                kwargs["val_loss"],
                kwargs["val_loss_segm"],
                kwargs["val_loss_bound"],
                kwargs["val_loss_dist"],
                segm_metrics["acc"],
                segm_metrics["mcc"],
                segm_metrics["kappa"],
                segm_metrics["precision"],
                segm_metrics["recall"],
                segm_metrics["iou"],
                segm_metrics["f1"],
                bound_metrics["acc"],
                bound_metrics["mcc"],
                bound_metrics["kappa"],
                bound_metrics["precision"],
                bound_metrics["recall"],
                bound_metrics["iou"],
                bound_metrics["f1"],
            ])

        improved = False
        if current["val_loss"] < best_metrics["val_loss"] - 1e-6:
            best_metrics["val_loss"] = current["val_loss"]
            improved = True

        for key in ("iou", "mcc", "recall", "f1"):
            if current[key] > best_metrics[key] + 1e-6:
                best_metrics[key] = current[key]
                improved = True 
        if improved:
            patience_counter = 0
            best_model_state = model.state_dict()
            best_epoch = epoch
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": best_model_state,
                    "optim_state": optimizer.state_dict(),
                    "best_metrics": best_metrics,
                },
                CKPT_DIR / "best_model.pth",
            )
            print(
                f"  ✓ saved best checkpoint | "
                f"val_loss={best_metrics['val_loss']:.4f}  "
                f"iou={best_metrics['iou']:.4f}  "
                f"mcc={best_metrics['mcc']:.4f}  "
                f"recall={best_metrics['recall']:.4f}  "
                f"f1={best_metrics['f1']:.4f}"
            )
        else:
            patience_counter += 1
            print(
                f"  ✗ no improvement in any metric "
                f"({patience_counter}/{patience})"
            )
 
        if patience_counter >= patience:
            print(f"\n[early stopping] triggered after {epoch + 1} epochs.")
            print(
                f"Best epoch: {best_epoch} | "
                f"val_loss={best_metrics['val_loss']:.4f}  "
                f"iou={best_metrics['iou']:.4f}  "
                f"mcc={best_metrics['mcc']:.4f}  "
                f"recall={best_metrics['recall']:.4f}  "
                f"f1={best_metrics['f1']:.4f}"
            )
            break
        save_checkpoint(
            resume_path, epoch, model, optimizer, scaler,
            best_metrics, patience_counter, best_epoch
        )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[info] Restored best model from epoch {best_epoch}")
    else:
        print("\n[info] No improvement recorded; keeping last model state.")
 
    final_path = CKPT_DIR / "final_model.pth"
    torch.save(
        {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_metrics": best_metrics,
        },
        final_path,
    )
    print(f"\nTraining completed in: {datetime.now() - start}")
    print(f"[log] Training log saved to: {log_path}")
    print(f"[log] Best model saved to: {CKPT_DIR / 'best_model.pth'}")
    print(f"[log] Final model saved to: {final_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PTAViT3D Training")
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--seed",          type=int,   default=48)
    parser.add_argument("--num_workers",   type=int,   default=2)
    parser.add_argument("--reuse_splits",  action="store_true", default=True)
    parser.add_argument("--block_size",    type=int,   default=512)
    parser.add_argument("--val_frac",      type=float, default=0.2)
    parser.add_argument("--patience",      type=int,   default=30)
    parser.add_argument("--split_dir",     type=str,   default="/home/ucl/elia/aryal/Single Input Model/4.S1/split_dir")
    parser.add_argument("--ckpt_dir",      type=str,   default="/home/ucl/elia/aryal/Single Input Model/4.S1")
    parser.add_argument("--zarr_root", type=str, default="/globalsc/ucl/elia/aryal/S1_zarr/descending")
    parser.add_argument("--years",     type=int, nargs="+", default=[2018, 2019, 2021])
    args = parser.parse_args()
    # args, _ = parser.parse_known_args()
    train(args)