import torch
import torch.nn as nn
import torch.nn.functional as F

def repeat(block, args, n, activation=None):
  return [layer for _ in range(n) for layer in (block(*args), activation) if layer]

def wn_conv(conv, wn, *args, **kwargs):
  return nn.utils.weight_norm(conv(*args, **kwargs)) if wn else conv(*args, **kwargs)

def wn_conv2d(wn, *args, **kwargs): return wn_conv(nn.Conv2d, wn, *args, **kwargs)
def wn_conv3d(wn, *args, **kwargs): return wn_conv(nn.Conv3d, wn, *args, **kwargs)

def wn_conv3dwrap(wn, nf, exp=0, lrr=0): return nn.Sequential(wn_conv3d(wn, nf, nf, 3, 1, 1), nn.ReLU(inplace=True))
def wn_conv2dwrap(wn, nf, exp=0, lrr=0): return nn.Sequential(wn_conv2d(wn, nf, nf, 3, 1, 1), nn.ReLU(inplace=True))

class wdsr_block(nn.Module):
  def __init__(self, conv, wn=True, nf=32, exp=6, lrr=0.8):
    super(wdsr_block, self).__init__()
    self.conv = nn.Sequential(
      conv(wn, nf, nf*exp, 1),
      nn.ReLU(inplace=True),
      conv(wn, nf*exp, int(nf*lrr), 1),
      conv(wn, int(nf*lrr), nf, 3, 1, 1)
    )

  def forward(self, x): return x + self.conv(x)

def wdsr3d_block(*args, **kwargs): return wdsr_block(wn_conv3d, *args, **kwargs)
def wdsr2d_block(*args, **kwargs): return wdsr_block(wn_conv2d, *args, **kwargs)

def upsample_conv2d(scale, nf=1, wn=True):
  return nn.Sequential(
    wn_conv2d(wn, nf, scale*scale, 3, 1, 1, padding_mode='reflect'),
    nn.ReLU(inplace=True),
    wn_conv2d(wn, scale*scale, scale*scale, 3, 1, 1, padding_mode='reflect'),
    torch.nn.PixelShuffle(scale)
  )

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class Model2DCommon(nn.Module):
  def __init__(self, res_block, upsample, scale=2, n_layers=8, n_filters=32, weight_norm=True, ksize=3, mean=0.0, std=1.0):
    super(Model2DCommon, self).__init__()
    self.register_buffer('mean', torch.tensor(mean))
    self.register_buffer('std', torch.tensor(std))
    self.upsample = upsample

    self.convPass = nn.Sequential(
      wn_conv2d(weight_norm, 1, n_filters, ksize, 1, 1), 
      nn.ReLU(inplace=True),
      *repeat(res_block, (weight_norm, n_filters), n_layers),
      wn_conv2d(weight_norm, n_filters, scale*scale, ksize, 1, 1),
      nn.PixelShuffle(scale)
    )

  def forward(self, x):
    x = (x - self.mean) / self.std
    x = self.upsample(x) + self.convPass(x)
    x = x * self.std + self.mean
    return x

class Model3DCommon(nn.Module):
  def __init__(self, res_block, upsample, scale=2, frames=7, n_layers=8, n_filters=32, weight_norm=True, ksize=3, mean=0.0, std=1.0):
    super(Model3DCommon, self).__init__()
    self.register_buffer('mean', torch.tensor(mean))
    self.register_buffer('std', torch.tensor(std))
    self.upsample = upsample

    bod2_cnt = (frames // (ksize - 1)) - 1
    self.convPass = nn.Sequential(
      wn_conv3d(weight_norm, 1, n_filters, ksize, 1, 1),
      nn.ReLU(inplace=True),
      *repeat(res_block, (weight_norm, n_filters), n_layers),
      *repeat(wn_conv3d, (weight_norm, n_filters, n_filters, ksize, 1, (0,1,1)), bod2_cnt, nn.ReLU(inplace=True)),
      wn_conv3d(weight_norm, n_filters, scale*scale, ksize, 1, (0,1,1)),
      Lambda(lambda x: x.squeeze(2)),
      nn.PixelShuffle(scale)
    )

  def forward(self, x):
    x = (x-self.mean) / self.std # normalize
    x = self.convPass(x) + self.upsample(x[:,:,3])
    x = x * self.std + self.mean # denormalize
    return x

def init_weights(m):
  if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

def make_model(name, upsample='bilinear', scale=2, **kwargs):
  if upsample == 'conv2d':
    upsample = upsample_conv2d(scale)
  elif upsample in ['bicubic', 'bilinear']:
    upsample = nn.Upsample(scale_factor=scale, mode=upsample, align_corners=False)
  else: raise ValueError

  if name=='3dwdsrnet':
    return Model3DCommon(wdsr3d_block, upsample, scale, **kwargs)
  elif name=='3dsrnet':
    return Model3DCommon(wn_conv3dwrap, upsample, scale, **kwargs)
  elif name=="2dwdsrnet":
    return Model2DCommon(wdsr2d_block, upsample, scale, **kwargs)
  elif name=="2dsrnet":
    return Model2DCommon(wn_conv2dwrap, upsample, scale, **kwargs)
  else: raise ValueError