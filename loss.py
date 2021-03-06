import torch
import torch.nn as nn
import torchvision

def psnr(x, y):
  mse = ((x - y) ** 2).mean(dim=(-3,-2,-1))
  psnr = -10 * torch.log10(mse + 1e-8)
  return psnr.mean()

class VggLoss(nn.Module):
  
  def __init__(self, mean=None, std=None, grayscale=True, add_l1=True):
    super(VggLoss, self).__init__()
 
    if not mean: mean = [0.485, 0.456, 0.406]
    if not std: std = [0.229, 0.224, 0.225]

    self.add_l1 = add_l1
    self.grayscale = grayscale
    self.register_buffer('mean', torch.Tensor(mean).view(1, 3, 1, 1))
    self.register_buffer('std', torch.Tensor(std).view(1, 3, 1, 1))

    cuts = [0, 4, 9]
    vgg_feat = torchvision.models.vgg16(pretrained=True).features
    self.blocks = nn.ModuleList(vgg_feat[cuts[i]: cuts[i+1]].eval() for i in range(4))
    for param in self.parameters():
      param.requires_grad = False

  def preprocess(self, x):
    if self.grayscale: 
      x = x.repeat(1, 3, 1, 1)
    x = (x - self.mean) / self.std
    return x

  def forward(self, pred, label):
    loss = 0.0
    if self.add_l1:
      loss += nn.functional.l1_loss(pred, label)

    pred = self.preprocess(pred)
    label = self.preprocess(label)

    for bl in self.blocks:
      pred = bl(pred)
      label = bl(label)
      loss += nn.functional.l1_loss(pred, label)
    return loss
