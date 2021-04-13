import torch

def psnr(x, y):
  EPS = 1e-8
  mse = torch.mean((x - y) ** 2, dim=(1,2,3))
  scores = -10 * torch.log10(mse + EPS)
  return torch.mean(scores, dim=0)