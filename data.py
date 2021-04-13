import torch
from torch.utils.data import Dataset
#from skimage import io, transform
import torchvision.transforms.functional as TF
import random
from PIL import Image
import os
from tqdm.auto import tqdm

def augment_init(do_hflip=False, do_vflip=False, do_crop=False, crop_size=None, img_size=None):
  do_hflip = do_hflip and random.random() > 0.5
  do_vflip = do_vflip and random.random() > 0.5

  if do_crop:
    crop_y = random.randint(0, img_size[0] - crop_size)
    crop_x = random.randint(0, img_size[1] - crop_size)

  return (do_hflip, do_vflip, do_crop, crop_y, crop_x, crop_size)

def augment(img, params, cs=1):
  do_hflip, do_vflip, do_crop, crop_y, crop_x, crop_size = params

  if do_crop: img = TF.crop(img, cs*crop_y, cs*crop_x, cs*crop_size, cs*crop_size)
  if do_hflip: img = TF.hflip(img)
  if do_vflip: img = TF.vflip(img)

  return img

class ImageSequenceDataset(Dataset):

  def __init__(self, basedir, example_name, label_name, dirs, do_grayscale=False, scale=2, sequence_len=7, img_offset=0, train=True, cache=False, separate_central_image=False, single_image=False, augmentation=None):
    for x, y in locals().items():
      if x != 'self':
        setattr(self, x, y)

    if cache: self.cache_all()

  def cache_all(self):
    self.images = [self.load_datapoint(dir) for dir in tqdm(self.dirs)]

  def apply_transforms(self, img, augment_params=None, crop_scale=1):
    # if self.augment:
    #   augment(img, augment_params, crop_scale)

    if self.do_grayscale: img = TF.to_grayscale(img)
    return TF.to_tensor(img)

  def load_datapoint(self, dir):
    abstract_path = os.path.join(self.basedir, dir, self.example_name)
    example_paths = [abstract_path.format(id) for id in range(self.img_offset, self.sequence_len+self.img_offset)]
    label_path = os.path.join(self.basedir, dir, self.label_name)

    if self.single_image:
      img = Image.open(exmaple_paths[self.sequence_len//2])
      augment_params = None
      X = self.apply_transforms(img, augment_params, 1)

    else:
      imgs = [Image.open(path) for path in example_paths]
      img_size = imgs[0].size

      # augment_params = augment_init(self.rand_hflip, self.rand_vflip, self.rand_crop, self.crop_size, img_size)
      augment_params = None

      imgs = [self.apply_transforms(img, augment_params, 1) for img in imgs]
      X = torch.stack(imgs)
      X = X.permute(1,0,2,3)

    Y = Image.open(label_path)
    Y = self.apply_transforms(Y, augment_params, self.scale)

    if self.separate_central_image and not self.single_image:
      return X, imgs[self.sequence_len//2], Y
    else:
      return X, Y

  # returns dataset size
  def __len__(self):
    return len(self.dirs)

  # returns the idx item of the dataset
  def __getitem__(self, idx):
    if self.cache: return self.images[idx]
    else: return self.load_datapoint(self.dirs[idx])


class Vimeo7GrayDataset(ImageSequenceDataset):

  def __init__(self, basedir, scale=2, sequence_len=7, train=True, **kwargs):

    kwargs['img_offset'] = 1 + ((7 - sequence_len) // 2)
    kwargs['example_name'] = "im{0}.jpg"
    kwargs['label_name'] = "label_x{0}.jpg".format(scale)
    kwargs['dirs'] = self.load_dirs(basedir, train)
    kwargs['basedir'] = os.path.join(basedir, 'sequences')
    kwargs['do_grayscale'] = False

    super().__init__(scale=scale, sequence_len=sequence_len, train=train, **kwargs)

  def load_dirs(self, basedir, train):
    filename = 'sep_trainlist.txt' if train else 'sep_testlist.txt'
    path = os.path.join(basedir, filename)
    with open(path) as f:
      dirlist = [s.strip() for s in f.readlines()]
    return dirlist

class FaceGrayDataset(ImageSequenceDataset):

  def __init__(self, basedir, scale=2, sequence_len=7, train=True, **kwargs):

    subdir = "train" if train else "test"

    kwargs['img_offset'] = 1 + ((7 - sequence_len) // 2)
    kwargs['example_name'] = "img_{0}.JPG"
    kwargs['label_name'] = "label.JPG" if scale==2 else "label_{0}.JPG".format(scale*32)
    kwargs['dirs'] = self.load_dirs(basedir, subdir)
    kwargs['basedir'] = os.path.join(basedir, subdir)
    kwargs['do_grayscale'] = True

    super().__init__(scale=scale, sequence_len=sequence_len, train=train, **kwargs)

  def load_dirs(self, basedir, subdir):
    return os.listdir(os.path.join(basedir, subdir))