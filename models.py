import torch.nn as nn
import torch.nn.functional as F

def repeat(block, args, activation, n):
  layers = []
  for _ in range(n):
    layers.append(block(*args))
    if activation: layers.append(activation)
  return layers

class Model3DSRnet(nn.Module):

  def __init__(self, scale=2, frames=7, n_layers=3, n_chan=64, ksize=3):
    super(Model3DSRnet, self).__init__()
    self.scale = scale
    activation = nn.ReLU(inplace=True)

    head = nn.Conv3d(1, n_chan, ksize, 1, 1)
    bod1 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, 1), activation, n_layers);

    # at every convolution the depth shrinks by k_size - 1
    bod2_cnt = (frames // (ksize - 1)) - 1
    bod2 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, (0,1,1)), activation, bod2_cnt);
    tail = nn.Conv3d(n_chan, scale*scale, ksize, 1, (0,1,1))

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

class Model3DSRnet2(nn.Module):

  def __init__(self, scale=2, frames=7, n_chan=64, ksize=3):
    super(Model3DSRnet, self).__init__()
    self.scale = scale

    head = nn.Conv3d(1, n_chan, ksize, 1, 1)
    bod1 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, 1), nn.ReLU, 3);

    # at every convolution the depth shrinks by k_size - 1
    bod2_cnt = (frames // (ksize - 1)) - 1
    bod2 = repeat(nn.Conv3d, (n_chan, n_chan, ksize, 1, (0,1,1)), nn.ReLU, bod2_cnt);
    tail = nn.Conv3d(n_chan, scale*scale, ksize, 1, (0,1,1))

    self.convPass = nn.Sequential(head, nn.ReLU(inplace=True), *bod1, *bod2, tail)
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