import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.utils import weight_norm

def repeat(block, args, n, activation=None):
  layers = []
  for _ in range(n):
    layers.append(block(*args))
    if activation: layers.append(activation)
  return layers

def wn_conv3d(inc, outc, ksize, stride=1, padding=0, weight_norm=True, padding_mode='zeros'):
  module = nn.Conv3d(inc, outc, ksize, stride, padding, padding_mode=padding_mode)
  return nn.utils.weight_norm(module) if weight_norm else module

class wdsr_block(nn.Module):

  def __init__(self, n_filters, expansion, weight_norm=True, low_rank_ratio = 0.8):
    super(wdsr_block, self).__init__()

    self.conv = nn.Sequential(
      wn_conv3d(n_filters, n_filters * expansion, 1, weight_norm=weight_norm),
      nn.ReLU(inplace=True),
      wn_conv3d(n_filters * expansion, int(n_filters * low_rank_ratio), 1, weight_norm=weight_norm),
      wn_conv3d(int(n_filters * low_rank_ratio), n_filters, 3, padding=1, weight_norm=weight_norm)
    )

  def forward(self, x):
    return x + self.conv(x)



class Model3DWDSRnet(nn.Module):

  def __init__(self, scale=2, frames=7, n_layers=3, n_filters=32, expansion=1, weight_norm=True, ksize=3):
    super(Model3DWDSRnet, self).__init__()
    self.scale = scale

    relu = nn.ReLU(inplace=True)
    bod2_cnt = (frames // (ksize - 1)) - 1

    self.convPass = nn.Sequential(
        wn_conv3d(1, n_filters, ksize, 1, 1),
        relu,
        *repeat(wdsr_block, (n_filters, expansion), n_layers),
        *repeat(nn.Conv3d, (n_filters, n_filters, ksize, 1, (0,1,1)), bod2_cnt, relu),
        nn.Conv3d(n_filters, scale*scale, ksize, 1, (0,1,1)),
    )

    self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
  
  def forward(self, x):
    lr = x[:, :, 3] #picks the image in the middle of the video sequence @TODO: try to pass the middle image as a separate input

    x = self.convPass(x)
    #up = self.bicubic(lr)

    #reshapes the tensor, so that the scale*scale channels become sub-pixels of the high resolution image
    x = x.squeeze(dim=2) # pixel_shuffle expects 4D tensors so we remove the depth dimension that is == 1
    x = F.pixel_shuffle(x, self.scale) # reorders pixels

    y = self.upsample(lr)
    # adds together the residual path with the upscaled image
    x = x.add(up)
    
    return x

class Model3DSRnet(nn.Module):

  def __init__(self, scale=2, frames=7, n_layers=3, n_chan=64, ksize=3):
    super(Model3DSRnet, self).__init__()
    self.scale = scale
    relu = nn.ReLU(inplace=True)

    head = nn.Conv3d(1, n_chan, ksize, 1, 1)
    bod1 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, 1), n_layers, activation)

    # at every convolution the depth shrinks by k_size - 1
    bod2_cnt = (frames // (ksize - 1)) - 1
    bod2 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, (0,1,1)), bod2_cnt, activation)
    tail = nn.Conv3d(n_chan, scale*scale, ksize, 1, (0,1,1))

    self.convPass = nn.Sequential(
      nn.Conv3d(1, n_chan, ksize, 1, 1), relu,
      repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, 1), n_layers, activation)
    )

    self.convPass = nn.Sequential(head, activation, *bod1, *bod2, tail)
    #self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
  
  def forward(self, x):
    lr = x[:, :, 3] #picks the image in the middle of the video sequence @TODO: try to pass the middle image as a separate input
    up = F.interpolate(lr, scale_factor=self.scale, mode='bicubic', align_corners=False)

    x = self.convPass(x)

    #up = self.bicubic(lr)

    #reshapes the tensor, so that the scale*scale channels become sub-pixels of the high resolution image
    x = x.squeeze(dim=2) # pixel_shuffle expects 4D tensors so we remove the depth dimension that is == 1
    x = F.pixel_shuffle(x, self.scale) # reorders pixels

    # adds together the residual path with the upscaled image
    x = x.add(up)
    
    return x

