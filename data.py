import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image

from tqdm.auto import tqdm

from glob import glob
import random
import os

import utils


class ImageSequenceDataset(Dataset):
  def __init__(self, basedir, xnames, yname, dirs, single_image=False, grayscale=True, cache=False):
    for x, y in locals().items():
      if x != 'self': 
        setattr(self, x, y)
    if cache: self.data = [self.load_datapoint(dir) for dir in tqdm(dirs)]

  def load_image(self, dir, name):
    img = Image.open(os.path.join(self.basedir, dir, name))
    if not self.grayscale: img = TF.to_grayscale(img)
    return TF.to_tensor(img)

  def load_datapoint(self, dir):
    if self.single_image: X = self.load_image(dir, self.xnames[3])
    else: X = torch.stack([self.load_image(dir, name) for name in self.xnames]).permute(1,0,2,3)
    Y = self.load_image(dir, self.yname)
    return X, Y

  def __len__(self): 
    return len(self.dirs)

  def __getitem__(self, idx):
    if self.cache: return self.data[idx]
    else: return self.load_datapoint(self.dirs[idx])

def VimeoDataset(basedir, scale=2, single_image=False, train=True, patches=True, cache=False):
  ext = 'png' if patches else 'jpg'
  xnames = ["im{}.{}".format(i+1, ext) for i in range(7)]
  yname = "label_x{}.{}".format(scale, ext)
  dirs = utils.readlines(os.path.join(basedir, 'sep_trainlist.txt' if train else 'sep_testlist.txt'))
  if patches: dirs = [os.path.join(dir, str(patch)) for dir in dirs for patch in range(8)]
  basedir = os.path.join(basedir, 'sequences')
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=True, cache=cache)

def VimeoSmallDataset(basedir, scale=2, single_image=False, train=True, cache=False):
  xnames = ["im{}.jpg".format(i+1) for i in range(7)]
  yname = "label_x{}.jpg".format(scale)
  dirs = sorted(list(glob(os.path.join(basedir,'*','*'))))
  basedir = '.'
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=True, cache=cache)

def FaceDataset(basedir, scale=2, single_image=False, train=False, cache=False):
  xnames = ["img_{}.JPG".foramt(i+1) for i in range(7)]
  yname = "label.JPG"
  subdir = 'train' if train else 'test'
  dirs = sorted(os.listdir(os.path.join(basedir, subdir)))
  basedir = os.path.join(basedir, subdir)
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=False, cache=cache)

def EyesDataset(basedir, scale=2, single_image=False, train=False, cache=False):
  xnames = ["im{}x{}.jpg".format(i+1, scale) for i in range(7)]
  yname = "im4.jpg"
  dirs = sorted(list(glob(os.path.join(basedir,'*','*'))))
  basedir = '.'
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=True, cache=cache)

def Eyes2Dataset(basedir, scale=2, single_image=False, train=False, cache=False):
  xnames = ["im{}_x{}.png".format(i+1, scale) for i in range(7)]
  yname = "label.png"
  dirs = sorted(list(glob(os.path.join(basedir,'*'))))
  basedir = '.'
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=True, cache=cache)

def TestDataset(basedir, scale=2, single_image=False, train=True, ext='png', cache=False):
  xnames = ["im{}.{}".format(i+1, ext) for i in range(7)]
  yname = "label_x{}.{}".format(scale, ext)
  dirs = sorted(list(glob(os.path.join(basedir,'*','*'))))
  basedir = '.'
  return ImageSequenceDataset(basedir, xnames, yname, dirs, single_image, grayscale=True, cache=cache)

def make_dataset(name, **kwargs):
  if name == "vimeo":
    return VimeoDataset(patches=False, **kwargs)
  elif name == 'vimeo_patches':
    return VimeoDataset(patches=True, **kwargs)
  elif name == 'vimeo_small':
    return VimeoSmallDataset(**kwargs)
  elif name == 'mlfdb':
    return FaceDataset(**kwargs)
  elif name == 'eyes':
    return EyesDataset(**kwargs)
  elif name == 'eyes2':
    return Eyes2Dataset(**kwargs)
  elif name == 'test':
    return TestDataset(**kwargs)
  else: raise ValueError

def make_loader(dataset, dataloader):
  dataset_obj = make_dataset(**dataset)
  return torch.utils.data.DataLoader(dataset_obj, **dataloader)