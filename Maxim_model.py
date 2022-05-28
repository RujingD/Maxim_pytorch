# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for the MAXIM model."""

from typing import Any, Sequence, Tuple
import torch
import einops
import ml_collections
import numpy as _np
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Conv1x1(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=0,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)
class Conv3x3(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)
class ConvT_up(nn.Module):  
    def __init__(self, in_channels, out_channels, pad=0,bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)
class Conv_down(nn.Module):  
    def __init__(self, in_channels, out_channels, pad=1,bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)



def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, channels,height, width = x.shape
  x=x.permute(0,2,3,1)      
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]

  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x.permute(0,3,1,2)

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x=x.permute(0,2,3,1) 
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x.permute(0,3,1,2)

class MlpBlock(nn.Module):   
  """A 1-hidden-layer MLP block, applied over the last dimension."""
  def __init__(self,in_channel,mlp_dim,dropout_rate=0.0,bias=True):
    super().__init__()
    self.in_channel=in_channel
    self.mlp_dim = mlp_dim
    self.bias = bias
    self.dropout_rate = dropout_rate
    self.linear1=nn.Linear(self.in_channel,self.mlp_dim,bias=self.bias)
    self.linear2=nn.Linear(self.mlp_dim,self.in_channel,bias=self.bias)
  
  def forward(self, x):
    x = einops.rearrange( x, "n c h w-> n h w c")
    x = self.linear1(x)
    x = F.gelu(x)
    x = F.dropout(x,p=self.dropout_rate) 
    x = self.linear2(x)
    return einops.rearrange( x, "n h w c -> n c h w")

class UpSampleRatio(nn.Module):   
  """Upsample features given a ratio > 0."""
  def __init__(self,in_channel,features,ratio,bias=True):
    super().__init__()
    self.in_channel=in_channel
    self.features=features
    self.ratio=ratio
    self.bias=bias 
    self.conv1=Conv1x1(self.in_channel,self.features,bias=self.bias)
  
  def forward(self, x):
    n,c,h,w = x.shape
    x_resize=torchvision.transforms.Resize( (int(h * self.ratio), int(w * self.ratio)),interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    x = x_resize(x)  
    x=self.conv1(x)
    return x

class CALayer(nn.Module):    
  """Squeeze-and-excitation block for channel attention.

    ref: https://arxiv.org/abs/1709.01507
  """
  def __init__(self,in_channel,features,reduction=4,bias=True): 
    super().__init__()
    self.in_channel=in_channel
    self.features=features
    self.reduction=reduction
    self.bias=bias
    self.conv_1=Conv1x1(self.in_channel,self.features//self.reduction, bias=self.bias)
    self.conv_2=Conv1x1(self.features//self.reduction,self.features, bias=self.bias)
  def forward(self, x):
    # 2D global average pooling
    y = torch.mean(x, dim=[2,3],keepdim=True)  
    # Squeeze (in Squeeze-Excitation)
    y = self.conv_1(y)
    y_fun1 = nn.ReLU()
    y=y_fun1(y)
    # Excitation (in Squeeze-Excitation)
    y=self.conv_2(y)
    y_fun2 = nn.Sigmoid()
    y=y_fun2(y)
    return x * y 


class RCAB(nn.Module):        
  """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
  def __init__(self,features,reduction=4,lrelu_slope=0.2,bias=True):
    super().__init__()
    self.features=features
    self.reduction=reduction
    self.lrelu_slope=lrelu_slope
    self.bias=bias
    self.conv3_1=Conv3x3(in_channels=self.features,out_channels=self.features,bias=self.bias)   
    self.conv3_2=Conv3x3(in_channels=self.features,out_channels=self.features,bias=self.bias)
    self.calayer=CALayer(in_channel=self.features,features=self.features, reduction=self.reduction,bias=self.bias)
  
  def forward(self, x):
    shortcut = x
    n,c,h,w=x.shape
    x=nn.LayerNorm([c,h,w])(x)
    x = self.conv3_1(x)    
    x = nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)  
    x = self.conv3_2(x)
    x = self.calayer(x)
    return x + shortcut    


class GridGatingUnit(nn.Module):  
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the second last.
  If applied on other dims, you should swapaxes first.
  """
  def __init__(self,h_size,bias=True):   
    super().__init__()
    self.h_size=h_size
    self.bias=bias
    self.linear=nn.Linear(self.h_size,self.h_size,bias=self.bias)
  def forward(self, x): 
    u,v = _np.split(x, 2,axis=1)     
    v_b, v_c, v_h, v_w = v.shape
    v_fun = nn.LayerNorm([v_c, v_h, v_w])
    v=v_fun(v)
    v = torch.swapaxes(v, -1, -2)  
    v = self.linear(v) 
    v = _np.swapaxes(v, -1, -2)
    return u * (v + 1.)


class GridGmlpLayer(nn.Module):   
  """Grid gMLP layer that performs global mixing of tokens."""
  def __init__(self,in_channel,grid_size: Sequence[int],bias = True,factor = 2,dropout_rate= 0.0):
    super().__init__()
    self.in_channel=in_channel    
    self.grid_size= grid_size
    self.bias=bias
    self.factor=factor
    self.dropout_rate=dropout_rate
    self.linear1=nn.Linear(self.in_channel,self.in_channel* self.factor, bias=self.bias)
    self.linear2=nn.Linear(self.in_channel,self.in_channel, bias=self.bias)
    self.gridgatingunit=GridGatingUnit(self.grid_size[0]*self.grid_size[1])       
  
  def forward(self, x, deterministic=True):
    n, num_channels,h, w = x.shape      
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw 
    x = block_images_einops(x, patch_size=(fh, fw))  
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    _n, _num_channels,_h, _w = x.shape
    y = nn.LayerNorm([_num_channels,_h,_w])(x)     
    y=torch.swapaxes(y, -1, -3)   
    y = self.linear1(y)
    y=torch.swapaxes(y, -1, -3)
    y = F.gelu(y)
    
    y = self.gridgatingunit(y) 
    
    y=torch.swapaxes(y, -1, -3) 
    y = self.linear2(y)
    y=torch.swapaxes(y, -1, -3)  
    y = F.dropout(y,self.dropout_rate,deterministic)   
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x


class BlockGatingUnit(nn.Module):   
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  def __init__(self,w_size,bias=True):
    super().__init__()
    self.w_size=w_size    
    self.bias=bias
    self.linear=nn.Linear(self.w_size, self.w_size,bias=self.bias)
  
  def forward(self, x):
    u, v = _np.split(x, 2, axis=1)
    v = nn.LayerNorm([v.shape[1],v.shape[2],v.shape[3]])(v)
    v = self.linear(v)   
    return u * (v + 1.)


class BlockGmlpLayer(nn.Module):  
  """Block gMLP layer that performs local mixing of tokens."""
  def __init__(self,in_channel,block_size,bias=True,factor=2,dropout_rate=0.0):
    super().__init__()
    self.in_channel=in_channel
    self.block_size=block_size
    self.factor=factor
    self.dropout_rate=dropout_rate
    self.bias=bias
    self.linear1=nn.Linear(self.in_channel,self.in_channel* self.factor, bias=self.bias) 
    self.linear2=nn.Linear(self.in_channel,self.in_channel, bias=self.bias)
    self.blockgatingunit=BlockGatingUnit(self.block_size[0]*self.block_size[1])  
  
  def forward(self, x, deterministic=True):
    n,num_channels,h,w = x.shape
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    x = block_images_einops(x, patch_size=(fh, fw))
    # MLP2: Local (block) mixing part, provides within-block communication.
    y = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3]])(x)
    y=torch.swapaxes(y,-1,-3)
    y = self.linear1(y)
    y=torch.swapaxes(y,-1,-3)
    y = F.gelu(y)
    y = self.blockgatingunit(y)
    y=torch.swapaxes(y,-1,-3)
    y = self.linear2(y)
    y=torch.swapaxes(y,-1,-3)
    y = F.dropout(y,self.dropout_rate,deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):   
  """The multi-axis gated MLP block."""
  def __init__(self, in_channel,block_size,grid_size,block_gmlp_factor= 2,grid_gmlp_factor = 2,input_proj_factor = 2,
  bias = True,dropout_rate = 0.0):
    super().__init__()
    self.in_channel=in_channel   
    self.grid_size=grid_size
    self.block_size=block_size
    self.block_gmlp_factor= block_gmlp_factor
    self.grid_gmlp_factor = grid_gmlp_factor
    self.input_proj_factor = input_proj_factor
    self.bias = bias
    self.dropout_rate =dropout_rate
    self.linear1=nn.Linear(self.in_channel,self.in_channel*self.input_proj_factor,bias=self.bias)
    self.gridgmlpLayer=GridGmlpLayer(in_channel=self.in_channel,grid_size=self.grid_size,bias=self.bias,
                      factor=self.grid_gmlp_factor,dropout_rate=self.dropout_rate)
    self.blockgmlpLayer=BlockGmlpLayer(in_channel=self.in_channel,block_size=self.block_size,bias=self.bias,
                      factor=self.block_gmlp_factor,dropout_rate=self.dropout_rate)
    self.linear2=nn.Linear(self.in_channel*self.input_proj_factor,self.in_channel,bias=self.bias)
  def forward(self, x, deterministic=True):
    shortcut = x
    n, num_channels,h, w,  = x.shape
    x = nn.LayerNorm([num_channels,h, w])(x)
    x= torch.swapaxes(x,-1,-3)
    x = self.linear1(x)  
    x =torch.swapaxes(x,-1,-3)
    
    x = F.gelu(x)
    u, v = _np.split(x, 2, axis=1)
    # GridGMLPLayer
    u = self.gridgmlpLayer(u)

    # BlockGMLPLayer
    v = self.blockgmlpLayer(v)
    x = torch.cat([u, v], dim=1)  
    x= torch.swapaxes(x,-1,-3)
    x =self.linear2(x)
    x = torch.swapaxes(x,-1,-3)
    x = F.dropout(x,self.dropout_rate,deterministic)
    x = x + shortcut
    return x


class RDCAB(nn.Module):     
  """Residual dense channel attention block. Used in Bottlenecks."""
  def __init__(self,in_channel,features,reduction= 16,bias= True,dropout_rate = 0.0):  
    super().__init__()
    self.in_channel=in_channel
    self.features = features
    self.reduction = reduction
    self.bias=bias
    self.dropout_rate=dropout_rate
    self.mlpblock=MlpBlock(in_channel=self.in_channel,mlp_dim=self.features,dropout_rate=self.dropout_rate,bias=self.bias)
    self.calayer=CALayer(in_channel=self.in_channel,features=self.features,reduction=self.reduction,bias=self.bias)
  
  def forward(self, x, deterministic=True):
    y = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3]])(x)
    y =  self.mlpblock(y)   
    y =  self.calayer(y)
    x = x + y
    return x


class BottleneckBlock(nn.Module):   
  """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""
  def __init__(self,in_channel,features,grid_size,block_size,num_groups=1,block_gmlp_factor=2,grid_gmlp_factor=2,input_proj_factor=2,
  channels_reduction=4,dropout_rate=0.0,bias=True):
    super().__init__()
    self.in_channel=in_channel
    self.features=features
    self.block_size=block_size
    self.grid_size=grid_size
    self.num_groups=num_groups
    self.block_gmlp_factor=block_gmlp_factor
    self.grid_gmlp_factor=grid_gmlp_factor
    self.input_proj_factor=input_proj_factor
    self.channels_reduction=channels_reduction
    self.dropout_rate=dropout_rate
    self.bias=bias
    self.conv1=Conv1x1(self.in_channel,self.features, bias=self.bias)
    for idx in range(self.num_groups):
      RSHMAG = ResidualSplitHeadMultiAxisGmlpLayer(
          in_channel=self.in_channel,
          grid_size=self.grid_size,
          block_size=self.block_size,
          block_gmlp_factor=self.block_gmlp_factor,
          grid_gmlp_factor=self.grid_gmlp_factor,
          input_proj_factor=self.input_proj_factor,
          bias=self.bias,
          dropout_rate=self.dropout_rate)
      setattr(self,f"RSHMAG_{idx}",RSHMAG)
      Rdcab = RDCAB(
          in_channel=self.in_channel,
          features=self.features,
          reduction=self.channels_reduction,
          bias=self.bias,
          dropout_rate=self.dropout_rate)
      setattr(self,f"Rdcab_{idx}",Rdcab)
  
  def forward(self, x):
    """Applies the Mixer block to inputs."""
    assert x.ndim == 4  # Input has shape [batch, c,h, w]
    # input projection
    x = self.conv1(x)
    shortcut_long = x

    for i in range(self.num_groups):
      RSHMAG=getattr(self,f"RSHMAG_{i}")
      x = RSHMAG(x)
      # Channel-mixing part, which provides within-patch communication.
      Rdcab=getattr(self,f"Rdcab_{i}")
      x =  Rdcab(x)
    # long skip-connect
    x = x + shortcut_long
    return x


class UNetEncoderBlock(nn.Module):   
  """Encoder block in MAXIM."""
  def __init__(self,in_channel,in_channel_bridge,features,block_size,grid_size,num_groups = 1,lrelu_slope = 0.2,block_gmlp_factor = 2,
  grid_gmlp_factor = 2,input_proj_factor= 2,channels_reduction= 4,dropout_rate= 0.0,downsample = True,use_global_mlp = True,
  bias = True,use_cross_gating = False):
    super().__init__()
    self.in_channel=in_channel
    self.in_channel_bridge=in_channel_bridge
    self.features=features
    self.block_size=block_size
    self.grid_size=grid_size
    self.num_groups = num_groups
    self.lrelu_slope = lrelu_slope
    self.block_gmlp_factor = block_gmlp_factor
    self.grid_gmlp_factor = grid_gmlp_factor
    self.input_proj_factor= input_proj_factor
    self.channels_reduction= channels_reduction
    self.dropout_rate= dropout_rate
    self.downsample = downsample
    self.use_global_mlp = use_global_mlp
    self.bias = bias
    self.use_cross_gating = use_cross_gating
    self.conv1=Conv1x1(self.in_channel+self.in_channel_bridge, self.features, bias=self.bias) 
    self.conv_down = Conv_down(self.features,self.features, bias=self.bias) 
    for idx in range(self.num_groups):
      if self.use_global_mlp:
        RSHMAG = ResidualSplitHeadMultiAxisGmlpLayer(
            in_channel=self.features, 
            block_size=self.block_size,
            grid_size=self.grid_size,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            bias=self.bias,
            dropout_rate=self.dropout_rate)
        setattr(self,f"RSHMAG_{idx}",RSHMAG)
      Rcab= RCAB(
          features=self.features,
          reduction=self.channels_reduction,
          lrelu_slope=self.lrelu_slope,
          bias=self.bias)
      setattr(self,f"Rcab_{idx}",Rcab)
    
    self.CGB = CrossGatingBlock(
        in_channel_x=self.features,
        in_channel_y=self.features,     
        features=self.features,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        input_proj_factor=self.input_proj_factor,
        upsample_y=False,
        bias=self.bias)
  def forward(self, x: _np.ndarray, skip: _np.ndarray = None,   
               enc: _np.ndarray = None, dec: _np.ndarray = None, *,
               deterministic: bool = True) -> _np.ndarray:
    if skip is not None:
      x = torch.cat([x, skip], dim=1)   
    x =self.conv1(x)
    shortcut_long = x
    for i in range(self.num_groups):
      if self.use_global_mlp:
        RSHMAG=getattr(self,f"RSHMAG_{i}")
        x = RSHMAG(x)
      Rcab=getattr(self,f"Rcab_{i}")
      x = Rcab(x)
    x = x + shortcut_long

    if enc is not None and dec is not None:
      assert self.use_cross_gating
      x, _ = self.CGB(x, enc + dec)

    if self.downsample:
      x_down = self.conv_down(x)
      return x_down, x
    else:
      return x


class UNetDecoderBlock(nn.Module):
  """Decoder block in MAXIM."""
  def __init__(self,in_channel,in_channel_bridge,features,grid_size,block_size,num_groups = 1,lrelu_slope = 0.2,block_gmlp_factor = 2,
  grid_gmlp_factor = 2,input_proj_factor = 2,channels_reduction = 4,dropout_rate = 0.0,downsample = True,
  use_global_mlp = True,bias = True):
    super().__init__()
    self.in_channel=in_channel
    self.in_channel_bridge=in_channel_bridge
    self.features=features
    self.grid_size=grid_size
    self.block_size=block_size
    self.num_groups = num_groups
    self.lrelu_slope = lrelu_slope
    self.block_gmlp_factor = block_gmlp_factor
    self.grid_gmlp_factor = grid_gmlp_factor
    self.input_proj_factor = input_proj_factor
    self.channels_reduction = channels_reduction
    self.dropout_rate = dropout_rate
    self.downsample = downsample
    self.use_global_mlp = use_global_mlp
    self.bias = bias
    self.convt_up=ConvT_up(self.in_channel,self.features,bias=self.bias)
    self.UNEB=UNetEncoderBlock(
        in_channel=self.features,
        in_channel_bridge=self.in_channel_bridge,
        features=self.features,
        block_size=self.block_size,
        grid_size=self.grid_size,
        num_groups=self.num_groups,
        lrelu_slope=self.lrelu_slope,
        block_gmlp_factor=self.block_gmlp_factor,
        grid_gmlp_factor=self.grid_gmlp_factor,
        input_proj_factor=self.input_proj_factor,
        channels_reduction=self.channels_reduction,
        dropout_rate=self.dropout_rate,
        downsample=False,
        use_global_mlp=self.use_global_mlp,
        bias=self.bias)
  
  def forward(self, x: _np.ndarray, bridge: _np.ndarray = None,
               deterministic: bool = True) -> _np.ndarray:
    
    x = self.convt_up(x)  # self.features 
    x = self.UNEB(x,skip=bridge,deterministic=deterministic)
    return x


class GetSpatialGatingWeights(nn.Module):  
  """Get gating weights for cross-gating MLP block."""
  def __init__(self,in_channel,block_size,grid_size,input_proj_factor=2,dropout_rate=0.0,bias=True):
    super().__init__()
    self.in_channel=in_channel
    self.block_size=block_size
    self.grid_size=grid_size
    self.input_proj_factor=input_proj_factor
    self.dropout_rate=dropout_rate
    self.bias=bias
    self.linear1=nn.Linear(self.in_channel,self.in_channel * self.input_proj_factor,bias=self.bias)
    self.linear2= nn.Linear(self.grid_size[0]*self.grid_size[1],self.grid_size[0]*self.grid_size[1], bias=self.bias)
    self.linear3= nn.Linear(self.block_size[0]*self.block_size[1],self.block_size[0]*self.block_size[1], bias=self.bias)
    self.linear4=nn.Linear(self.in_channel * self.input_proj_factor,self.in_channel,bias=self.bias)
  
  def forward(self, x):
    n, num_channels,h, w = x.shape

    # input projection
    x = nn.LayerNorm([num_channels,h,w])(x)
    x= torch.swapaxes(x,-1,-3)
    x = self.linear1(x)
    x= torch.swapaxes(x,-1,-3)
    x = F.gelu(x)
    u, v = _np.split(x, 2, axis=1)

    # Get grid MLP weights
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    u = block_images_einops(u, patch_size=(fh, fw))
    u = _np.swapaxes(u, -1, -2)
    u = self.linear2(u) 
    u = _np.swapaxes(u, -1, -2)

    u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))
    # Get Block MLP weights
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw

    v = block_images_einops(v, patch_size=(fh, fw))
    
    v = self.linear3(v)
    
    v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))
    x = torch.cat([u, v], dim=1)

    x = _np.swapaxes(x, -1, -3)
    x = self.linear4(x)
    x = _np.swapaxes(x, -1, -3)
    x = F.dropout(x,p=self.dropout_rate)
    return x


class CrossGatingBlock(nn.Module):   
  """Cross-gating MLP block."""
  def __init__(self,in_channel_x,in_channel_y,features,grid_size,block_size,dropout_rate = 0.0,input_proj_factor = 2,upsample_y = True,bias = True):
    super().__init__()
    self.in_channel_x=in_channel_x
    self.in_channel_y=in_channel_y 
    self.features=features
    self.grid_size=grid_size
    self.block_size=block_size
    self.dropout_rate = dropout_rate
    self.input_proj_factor = input_proj_factor
    self.upsample_y = upsample_y
    self.bias = bias
    self.convt_up=ConvT_up(self.in_channel_y,self.features, bias=self.bias)
    self.conv1_1=Conv1x1(self.in_channel_x,self.features, bias=self.bias)
    self.conv1_2=Conv1x1(self.features,self.features, bias=self.bias)
    self.linear1=nn.Linear(self.features,self.features, bias=self.bias)
    self.getspatialgatingweights1=GetSpatialGatingWeights(
        in_channel=self.features,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        bias=self.bias)
    self.linear2=nn.Linear(self.features,self.features, bias=self.bias)
    self.getspatialgatingweights2=GetSpatialGatingWeights(
        in_channel=self.features,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        bias=self.bias)
    self.linear3=nn.Linear(self.features,self.features, bias=self.bias)
    self.linear4=nn.Linear(self.features,self.features, bias=self.bias)
  
  def forward(self, x, y):
    # Upscale Y signal, y is the gating signal.
    if self.upsample_y:
      y = self.convt_up(y)
    x = self.conv1_1(x)
    y = self.conv1_2(y)
    assert y.shape == x.shape  #self.features
    shortcut_x = x
    shortcut_y = y

    # Get gating weights from X
    x = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3]])(x)
    x= _np.swapaxes(x, -1, -3)
    x = self.linear1(x)
    x = _np.swapaxes(x, -1, -3)
    x = F.gelu(x)
    gx = self.getspatialgatingweights1(x)

    # Get gating weights from Y
    y = nn.LayerNorm([y.shape[1],y.shape[2],y.shape[3]])(y)
    y= _np.swapaxes(y, -1, -3)
    y = self.linear2(y)
    y= _np.swapaxes(y, -1, -3)
    y = F.gelu(y)
    gy = self.getspatialgatingweights2(y)

    # Apply cross gating: X = X * GY, Y = Y * GX
    y = y * gx
    y=_np.swapaxes(y, -1, -3)
    y = self.linear3(y)
    y=_np.swapaxes(y, -1, -3)
    y = F.dropout(y,p=self.dropout_rate)
    y = y + shortcut_y

    x = x * gy  # gating x using y
    x=_np.swapaxes(x, -1, -3)
    x = self.linear4(x)
    x=_np.swapaxes(x, -1, -3)
    x = F.dropout(x,p=self.dropout_rate)
    x = x + y + shortcut_x  # get all aggregated signals
    return x, y


class SAM(nn.Module):  
  """Supervised attention module for multi-stage training.

    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """
  def __init__(self,in_channel,features,output_channels=3,bias=True):
    super().__init__()
    self.in_channel=in_channel     
    self.features=features
    self.output_channels=output_channels
    self.bias=bias
    self.conv3_1=Conv3x3(self.in_channel,self.features, bias=self.bias)
    self.conv3_2=Conv3x3(self.features,self.output_channels, bias=self.bias)
    self.conv3_3=Conv3x3(self.output_channels, self.features,bias=self.bias)
  
  def forward(self, x: _np.ndarray, x_image: _np.ndarray, *,
               train: bool) -> Tuple[_np.ndarray, _np.ndarray]:
    """Apply the SAM module to the input and features.

    Args:
      x: the output features from UNet decoder with shape (h, w, c)
      x_image: the input image with shape (h, w, 3)
      train: Whether it is training

    Returns:
      A tuple of tensors (x1, image) where (x1) is the sam features used for the
        next stage, and (image) is the output restored image at current stage.
    """
    # Get features
    x1 = self.conv3_1(x)
    # Output restored image X_s
    if self.output_channels == 3:
      image = self.conv3_2(x) + x_image
    else:
      image = self.conv3_2(x)
    # Get attention maps for features
    x2=self.conv3_3(image)
    x2_fun = nn.Sigmoid()   
    x2=x2_fun(x2)
    # Get attended feature maps
    x1 = x1 * x2
    # Residual connection
    x1 = x1 + x    
    return x1, image


class MAXIM(nn.Module):
  """The MAXIM model function with multi-stage and multi-scale supervision.

  For more model details, please check the CVPR paper:
  MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)

  Attributes:
    features: initial hidden dimension for the input resolution.
    depth: the number of downsampling depth for the model.
    num_stages: how many stages to use. It will also affects the output list.
    num_groups: how many blocks each stage contains.
    bias: whether to use bias in all the conv/mlp layers.
    num_supervision_scales: the number of desired supervision scales.
    lrelu_slope: the negative slope parameter in leaky_relu layers.
    use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
      layer.
    use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
      skip connections and multi-stage feature fusion layers.
    high_res_stages: how many stages are specificied as high-res stages. The
      rest (depth - high_res_stages) are called low_res_stages.
    block_size_hr: the block_size parameter for high-res stages.
    block_size_lr: the block_size parameter for low-res stages.
    grid_size_hr: the grid_size parameter for high-res stages.
    grid_size_lr: the grid_size parameter for low-res stages.
    num_bottleneck_blocks: how many bottleneck blocks.
    block_gmlp_factor: the input projection factor for block_gMLP layers.
    grid_gmlp_factor: the input projection factor for grid_gMLP layers.
    input_proj_factor: the input projection factor for the MAB block.
    channels_reduction: the channel reduction factor for SE layer.
    num_outputs: the output channels.
    dropout_rate: Dropout rate.

  Returns:
    The output contains a list of arrays consisting of multi-stage multi-scale
    outputs. For example, if num_stages = num_supervision_scales = 3 (the
    model used in the paper), the output specs are: outputs =
    [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
     [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
     [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
    The final output can be retrieved by outputs[-1][-1].
  """
  def __init__(self,in_channel,features= 64,depth = 3,num_stages = 2,num_groups = 1,use_bias=True,num_supervision_scales = 1,
  lrelu_slope= 0.2,use_global_mlp = True,use_cross_gating= True,high_res_stages= 2,block_size_hr = (16, 16),
  block_size_lr = (8, 8),grid_size_hr= (16, 16),grid_size_lr = (8, 8),num_bottleneck_blocks = 1,block_gmlp_factor= 2,
  grid_gmlp_factor= 2,input_proj_factor= 2,channels_reduction = 4,num_outputs = 3,dropout_rate = 0.0):
    super().__init__()
    self.in_channel=in_channel
    self.features= features
    self.depth = depth
    self.num_stages = num_stages
    self.num_groups = num_groups
    self.use_bias=use_bias
    self.num_supervision_scales = num_supervision_scales
    self.lrelu_slope= lrelu_slope
    self.use_global_mlp = use_global_mlp
    self.use_cross_gating= use_cross_gating    
    self.high_res_stages= high_res_stages
    self.block_size_hr = block_size_hr
    self.block_size_lr = block_size_lr
    self.grid_size_hr= grid_size_hr
    self.grid_size_lr = grid_size_lr
    self.num_bottleneck_blocks = num_bottleneck_blocks
    self.block_gmlp_factor= block_gmlp_factor
    self.grid_gmlp_factor= grid_gmlp_factor
    self.input_proj_factor= input_proj_factor
    self.channels_reduction = channels_reduction
    self.num_outputs = num_outputs
    self.dropout_rate = dropout_rate
    
    for idx_stage in range(self.num_stages):
      for i in range(self.num_supervision_scales):
        conv = Conv3x3(
            in_channels = 3,
            out_channels=(2**i) * self.features,
            bias=self.use_bias)
        setattr(self, f"conv_{i}",conv )
        if idx_stage > 0:
          if self.use_cross_gating:
            block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
            grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr
            CGB = CrossGatingBlock(
                    in_channel_x=self.features*(2**i),   
                    in_channel_y=self.features*(2**i),
                    features=(2**i) * self.features,
                    block_size=block_size,
                    grid_size=grid_size,
                    dropout_rate=self.dropout_rate,
                    input_proj_factor=self.input_proj_factor,
                    upsample_y=False,
                    bias=self.use_bias)
            setattr(self, f"CGB_{idx_stage}_{i}",CGB ) 
          else:
            if i==0:
              in_channel_tmp=64
            else:
              in_channel_tmp=(2**i) * self.features*2
            _tmp = Conv1x1(in_channel_tmp,(2**i) * self.features,bias=self.use_bias) 
            setattr(self, f"stage_{idx_stage}_input_catconv_{i}",_tmp ) 


      for i in range(self.depth):  
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr
        use_cross_gating_layer = True if idx_stage > 0 else False
        in_channel_skip=(2**i) * self.features  if i<self.num_supervision_scales else 0
        if i==0:
          in_channel_temp=self.features        
        else:
          in_channel_temp=(2**(i-1)) *self.features
        UEB= UNetEncoderBlock(
            in_channel=in_channel_temp+in_channel_skip,
            in_channel_bridge=0,
            features=(2**i) * self.features,
            num_groups=self.num_groups,
            downsample=True,
            lrelu_slope=self.lrelu_slope,
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,
            dropout_rate=self.dropout_rate,
            bias=self.use_bias,
            use_cross_gating=use_cross_gating_layer) 
        setattr(self, f"UEB{idx_stage}_{i}", UEB)

      for i in range(self.num_bottleneck_blocks):
        if i==0:
          in_channel_temp=(2**(self.depth-1)) * self.features
        else:
          in_channel_temp=(2**(self.depth - 1)) * self.features
        BLB = BottleneckBlock(
            in_channel=in_channel_temp,
            features=(2**(self.depth - 1)) * self.features,
            block_size=self.block_size_lr,
            grid_size=self.block_size_lr,
            num_groups=self.num_groups,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,
            dropout_rate=self.dropout_rate,
            bias=self.use_bias)
        setattr(self, f"BLB_{idx_stage}_{i}", BLB)
      
      for i in reversed(range(self.depth)):  
        # use larger blocksize at high-res stages
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr

        # get additional multi-scale signals
        for j in range(self.depth):
          in_channel_temp=(2**j)*self.features
          _UpSampleRatio = UpSampleRatio(
              in_channel=in_channel_temp, 
              features=(2**i) * self.features,
              ratio=2**(j - i),
              bias=self.use_bias)
          setattr(self,f"UpSampleRatio_{idx_stage}_{i}_{j}", _UpSampleRatio)
        
        # Use cross-gating to cross modulate features
        if self.use_cross_gating:
          if i==self.depth-1:
            in_channel_x_temp=384                      
            in_channel_y_temp=128
          else:
            in_channel_x_temp=in_channel_x_temp//2
            in_channel_y_temp=(2**(i+1)) * self.features
          _CrossGatingBlock = CrossGatingBlock(
              in_channel_x=in_channel_x_temp,
              in_channel_y=in_channel_y_temp,                 
              features=(2**i) * self.features,
              block_size=block_size,
              grid_size=grid_size,
              dropout_rate=self.dropout_rate,
              input_proj_factor=self.input_proj_factor,
              upsample_y=True,
              bias=self.use_bias)
          setattr(self,f"stage_{idx_stage}_cross_gating_block_{i}", _CrossGatingBlock)
        else:
          if i==self.depth-1:
            in_channel_x_temp=384                       
            in_channel_y_temp=64
          else:
            in_channel_x_temp=in_channel_x_temp//2
            in_channel_y_temp=(2**(i+1)) * self.features
          _tmp= Conv1x1( in_channel_x_temp,(2**i) * self.features, bias=self.use_bias) 
          setattr(self,f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv11_{i}",_tmp)
          _tmp = Conv3x3((2**i) * self.features,(2**i) * self.features, bias=self.use_bias)
          setattr(self,f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv33_{i}",_tmp)


      # start decoder. Multi-scale feature fusion of cross-gated features
      for i in reversed(range(self.depth)):  
        # use larger blocksize at high-res stages
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr

        for j in range(self.depth):
            if j==0:
              in_channel_temp=128                
            else:
              in_channel_temp=in_channel_temp//2      
            _UpSampleRatio = UpSampleRatio(
                in_channel=in_channel_temp, 
                features=(2**i) * self.features,
                ratio=2**(self.depth - j - 1 - i),
                bias=self.use_bias)
            setattr(self,f"UpSampleRatio_skip_signals_{idx_stage}_{i}_{j}", _UpSampleRatio)
        if i==self.depth-1:
          in_channel_temp_UDB=128
          in_channel_temp_UDB_skip=384
        else:
          in_channel_temp_UDB=(2**(i+1)) * self.features
          in_channel_temp_UDB_skip=in_channel_temp_UDB_skip//2
        _UNetDecoderBlock = UNetDecoderBlock(
          in_channel=in_channel_temp_UDB,                         
          in_channel_bridge=in_channel_temp_UDB_skip,
          features=(2**i) * self.features,
          grid_size=grid_size,
          block_size=block_size,
          num_groups=self.num_groups,
          lrelu_slope=self.lrelu_slope,
          block_gmlp_factor=self.block_gmlp_factor,
          grid_gmlp_factor=self.grid_gmlp_factor,
          input_proj_factor=self.input_proj_factor,
          channels_reduction=self.channels_reduction,
          dropout_rate=self.dropout_rate,
          use_global_mlp=self.use_global_mlp, 
          bias=self.use_bias) 
        setattr(self,f"stage_{idx_stage}_decoder_block_{i}",_UNetDecoderBlock)

        if i < self.num_supervision_scales:
          if idx_stage < self.num_stages - 1:  # not last stage, apply SAM
            _SAM = SAM(
              in_channel = (2**i) * self.features,              
              features=(2**i) * self.features,
              output_channels=self.num_outputs,
              bias=self.use_bias)
            setattr(self,f"stage_{idx_stage}_supervised_attention_module_{i}", _SAM)

          else:  # Last stage, apply output convolutions
            _Conv3x3 = Conv3x3((2**i) * self.features,self.num_outputs, bias=self.use_bias)
            setattr(self, f"stage_{idx_stage}_output_conv_{i}",_Conv3x3)



  
  def forward(self, x, *, train: bool = False) -> Any:
    n,c, h, w,  = x.shape  # input image shape
    shortcuts = []
    shortcuts.append(x)    
    # Get multi-scale input images
 
    for i in range(1, self.num_supervision_scales):  
      r=torchvision.transforms.Resize( size=(h//(2**i), w//(2**i)),interpolation=torchvision.transforms.InterpolationMode.NEAREST)
      shortcuts.append(r(x))
    outputs_all = []
    sam_features, encs_prev, decs_prev = [], [], []

    for idx_stage in range(self.num_stages):  
      # Input convolution, get multi-scale input features
      x_scales = []
      for i in range(self.num_supervision_scales):
        conv=getattr(self,f"conv_{i}")
        x_scale = conv(shortcuts[i])  
        # If later stages, fuse input features with SAM features from prev stage
        if idx_stage > 0: 
          # use larger blocksize at high-res stages
          if self.use_cross_gating:  
            CGB=getattr(self,f"CGB_{idx_stage}_{i}")
            x_scale, _ = CGB(x_scale, sam_features.pop())
          else:
            x_scale_temp=torch.cat([x_scale, sam_features.pop()], dim=1) 
            # print(i,x_scale_temp.shape,'x_scale_temp.shape')  
            x_scale =getattr(self, f"stage_{idx_stage}_input_catconv_{i}")\
                     (x_scale_temp)
              
        x_scales.append(x_scale)

      # start encoder blocks
      encs = []
      x = x_scales[0]  # First full-scale input feature
      for i in range(self.depth):  
        # use larger blocksize at high-res stages, vice versa.
        use_cross_gating_layer =True if idx_stage > 0 else False
        # Multi-scale input if multi-scale supervision
        x_scale = x_scales[i] if i < self.num_supervision_scales else None

        # UNet Encoder block
        enc_prev = encs_prev.pop() if idx_stage > 0 else None
        dec_prev = decs_prev.pop() if idx_stage > 0 else None
        UEB=getattr(self,f"UEB{idx_stage}_{i}")
        x, bridge = UEB(x,skip=x_scale,enc=enc_prev,dec=dec_prev)   
        # Cache skip signals
        encs.append(bridge)
      # Global MLP bottleneck blocks
      for i in range(self.num_bottleneck_blocks):   
        BLB=getattr(self,f"BLB_{idx_stage}_{i}")
        x = BLB(x)
      # cache global feature for cross-gating
      global_feature = x
      # start cross gating. Use multi-scale feature fusion
      skip_features = []

      
      for i in reversed(range(self.depth)):  
        # use larger blocksize at high-res stages
        # get additional multi-scale signals
        signal = torch.cat([
            getattr(self, f"UpSampleRatio_{idx_stage}_{i}_{j}")\
            (enc) for j, enc in enumerate(encs)],dim=1)
        # Use cross-gating to cross modulate features
        if self.use_cross_gating:
          skips, global_feature = getattr(self,f"stage_{idx_stage}_cross_gating_block_{i}")\
          (signal, global_feature)
        else:  
          
          skips = getattr(self,f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv11_{i}")\
          (signal)
         
          skips = getattr(self,f"stage_{idx_stage}_cross_gating_block_no_use_cross_gating_conv33_{i}")\
          (skips)
        skip_features.append(skips)
      # start decoder. Multi-scale feature fusion of cross-gated features
      outputs, decs, sam_features = [], [], []
      for i in reversed(range(self.depth)):
        # use larger blocksize at high-res stages
        # get multi-scale skip signals from cross-gating block
        
        signal = torch.cat([
          getattr(self,f"UpSampleRatio_skip_signals_{idx_stage}_{i}_{j}")(skip)
            for j, skip in enumerate(skip_features)],dim=1)
        # UNetDecoderBlock
        x= getattr(self,f"stage_{idx_stage}_decoder_block_{i}")(x,bridge=signal)
        # Cache decoder features for later-stage's usage
        decs.append(x)

        # output conv, if not final stage, use supervised-attention-block.
        if i < self.num_supervision_scales:
          if idx_stage < self.num_stages - 1:  # not last stage, apply SAM
            sam, output = getattr(self, f"stage_{idx_stage}_supervised_attention_module_{i}")\
            (x, shortcuts[i], train=train)
            outputs.append(output)
            sam_features.append(sam)
          else:  # Last stage, apply output convolutions
            output =getattr(self,f"stage_{idx_stage}_output_conv_{i}")(x)
            output = output + shortcuts[i]
            outputs.append(output)
      # Cache encoder and decoder features for later-stage's usage
      encs_prev = encs[::-1]
      decs_prev = decs

      # Store outputs
      outputs_all.append(outputs)
    return outputs_all


def Model(*, variant=None, **kw):
  """Factory function to easily create a Model variant like "S".

  Every model file should have this Model() function that returns the flax
  model function. The function name should be fixed.

  Args:
    variant: UNet model variants. Options: 'S-1' | 'S-2' | 'S-3'
        | 'M-1' | 'M-2' | 'M-3'
    **kw: Other UNet config dicts.

  Returns:
    The MAXIM() model function
  """

  if variant is not None:
    config = {
        # params: 6.108515000000001 M, GFLOPS: 93.163716608
        "S-1": {
            "features": 32,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 13.35383 M, GFLOPS: 206.743273472
        "S-2": {
            "features": 32,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 20.599145 M, GFLOPS: 320.32194560000005
        "S-3": {
            "features": 32,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 19.361219000000002 M, 308.495712256 GFLOPs
        "M-1": {
            "features": 64,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 40.83911 M, 675.25541888 GFLOPs
        "M-2": {
            "features": 64,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 62.317001 M, 1042.014666752 GFLOPs
        "M-3": {
            "features": 64,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
    }[variant]

    for k, v in config.items():
      kw.setdefault(k, v)

  return MAXIM(**kw)


 

