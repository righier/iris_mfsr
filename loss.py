import torch
import torch.nn as nn

def psnr(x, y):
  EPS = 1e-8
  mse = torch.mean((x - y) ** 2, dim=(1,2,3))
  scores = -10 * torch.log10(mse + EPS)
  return torch.mean(scores, dim=0)

class VggLoss(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], grayscale=True):
        super(VggLoss, self).__init__()
        
        self.grayscale = grayscale
        self.register_buffer('mean', torch.Tensor(mean).view(1,3,1,1))
        self.register_buffer('std', torch.Tensor(std).view(1,3,1,1))
        
        cuts = [0, 4, 9, 16, 23]
        vgg_feat = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList(vgg_feat[cuts[i]: cuts[i+1]].eval() for i in range(4))
        for param in self.parameters(): param.requires_grad = False
    
    def preprocess(self, x):
        if self.grayscale: x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return x
        
    def forward(self, pred, label):
        pred = self.preprocess(pred)
        label = self.preprocess(label)
        
        loss = 0.0
        for bl in self.blocks:
            pred = bl(pred)
            label = bl(label)
            loss += nn.functional.l1_loss(pred, label)
        return loss
