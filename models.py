import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.utils import weight_norm

def repeat(block, args, n, activation=None):
  layers = []
  for _ in range(n):
    layers.append(block(*args))
    if activation: layers.append(activation)
  return layers

def wn_conv2d(inc, outc, ksize, stride=1, padding=0, weight_norm=True, padding_mode='zeros'):
  module = nn.Conv2d(inc, outc, ksize, stride, padding, padding_mode=padding_mode)
  return nn.utils.weight_norm(module) if weight_norm else module

def wn_conv3d(inc, outc, ksize, stride=1, padding=0, weight_norm=True, padding_mode='zeros'):
  module = nn.Conv3d(inc, outc, ksize, stride, padding, padding_mode=padding_mode)
  return nn.utils.weight_norm(module) if weight_norm else module

class wdsr3d_block(nn.Module):
  def __init__(self, n_filters, expansion=6, weight_norm=True, low_rank_ratio = 0.8):
    super(wdsr3d_block, self).__init__()
    self.conv = nn.Sequential(
      wn_conv3d(n_filters, n_filters * expansion, 1, weight_norm=weight_norm), 
      nn.ReLU(inplace=True),
      wn_conv3d(n_filters * expansion, int(n_filters * low_rank_ratio), 1, weight_norm=weight_norm),
      wn_conv3d(int(n_filters * low_rank_ratio), n_filters, 3, padding=1, weight_norm=weight_norm)
    )

  def forward(self, x):
    return x + self.conv(x)

class wdsr2d_block(nn.Module):
  def __init__(self, n_filters, expansion=6, weight_norm=True, low_rank_ratio = 0.8):
    super(wdsr3d_block, self).__init__()
    self.conv = nn.Sequential(
      wn_conv2d(n_filters, n_filters * expansion, 1, weight_norm=weight_norm), 
      nn.ReLU(inplace=True),
      wn_conv2d(n_filters * expansion, int(n_filters * low_rank_ratio), 1, weight_norm=weight_norm),
      wn_conv2d(int(n_filters * low_rank_ratio), n_filters, 3, padding=1, weight_norm=weight_norm)
    )

  def forward(self, x):
    return x + self.conv(x)

class upsample_conv2d(nn.Module):
  def __init__(self, scale, n_filters=1):
    super(upsample_conv2d, self).__init__()
    self.scale = scale
    self.body = nn.Sequential(
      wn_conv2d(n_filters, scale*scale, 3, padding=1, padding_mode='reflect'),
      nn.ReLU(inplace=True),
      wn_conv2d(scale*scale, scale*scale, 3, padding=1, padding_mode='reflect'),
    )

  def forward(self, x):
    x = self.body(x)
    x = F.pixel_shuffle(x, self.scale)
    return x


class Model3DCommon(nn.Module):

  def __init__(self, res_block, upsample, scale=2, frames=7, n_layers=8, n_filters=32, expansion=6, weight_norm=True, ksize=3):
    super(Model3DCommon, self).__init__()
    self.scale = scale

    relu = nn.ReLU(inplace=True)
    bod2_cnt = (frames // (ksize - 1)) - 1

    self.convPass = nn.Sequential(
      wn_conv3d(1, n_filters, ksize, 1, 1),
      relu,
      *repeat(res_block, (n_filters, expansion, weight_norm), n_layers),
      *repeat(wn_conv3d, (n_filters, n_filters, ksize, 1, (0,1,1), weight_norm), bod2_cnt, relu),
      wn_conv3d(n_filters, scale*scale, ksize, 1, (0,1,1), weight_norm),
    )

    self.upsample = upsample
  
  def forward(self, x):
    lr = x[:, :, 3] #picks the image in the middle of the video sequence @TODO: try to pass the middle image as a separate input
    y = self.upsample(lr)

    x = self.convPass(x)
    x = x.squeeze(dim=2) # pixel_shuffle expects 4D tensors so we remove the depth dimension that is == 1
    x = F.pixel_shuffle(x, self.scale) # reorders pixels

    x = x.add(y)
    return x

def upsample_naive(scale, mode='bicubic'):
  return nn.Upsample(scale_factor=scale, mode=mode, align_corners=False)

def Model3DWDSRnet(res_block=wdsr3d_block, upsample=upsample_conv2d, scale=2, frames=7, n_layers=8, n_filters=32, expansion=6, weight_norm=True, ksize=3):
  upsample = upsample(scale)
  return Model3DCommon(res_block, upsample, scale, frames, n_layers, n_filters, expansion, weight_norm, ksize)

def Model3DSRnet(scale=2, frames=7, n_layers=4, n_filters=64, weight_norm=False, ksize=3):
  upsample = upsample_naive(scale)
  return Model3DCommon(res_block, upsample, scale, frames, n_layers, n_filters, expansion, weight_norm, ksize)
