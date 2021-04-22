from tqdm.auto import tqdm 
import wandb

import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.optim as optim

import loss
import utils
import models
import data

class Trainer():

  def __init__(self, save_path, device, model, trainloader, testloader, samples, epochs, optimizer, criterion, score, log_freq, eval_freq, mixed_precision, single_image):
    self.save_path = save_path
    self.device = device
    self.device_cpu = utils.get_cpu_device()
    self.model = model
    self.trainloader = trainloader
    self.testloader = testloader
    self.samples = samples
    self.epochs = epochs
    self.log_freq = log_freq
    self.eval_freq = eval_freq
    self.mixed_precision = mixed_precision
    self.single_image = single_image

    self.init_criterion(criterion)
    self.init_score(score)
    self.init_optimizer(**optimizer)

    if eval_freq > len(self.trainloader): 
      self.eval_freq = len(self.trainloader)

  def init_criterion(self, criterion):
    if criterion=='l1': self.criterion = nn.L1Loss().to(self.device)
    elif criterion=='mse': self.criterion = nn.MSELoss().to(self.device)
    elif criterion=='vgg': self.criterion = loss.VggLoss(add_l1=False).to(self.device)
    elif criterion=='vgg-l1': self.criterion = loss.VggLoss(add_l1=True).to(self.device)
    else: raise ValueError

  def init_score(self, score):
    if score=='psnr': self.score = loss.psnr
    else: raise ValueError

  def init_optimizer(self, name, lr, weight_decay, scheduler):
    if name=="adam":    self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name=="adamw": self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise ValueError
    self.init_scheduler(**scheduler)

  def init_scheduler(self, name, max_lr):
    if name=='onecycle':
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr, epochs=self.epochs, steps_per_epoch=len(self.trainloader))
      self.freq_scheduler = True ## True if it needs to update every mini batch
    elif name=='reduce_on_plateau':
      self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
      self.freq_scheduler = False
    else: raise ValueError

  def test_batch(self, X, Y):
    X, Y = X.to(self.device), Y.to(self.device)
    outputs = self.model(X)
    loss = float(self.criterion(outputs, Y))
    score = float(self.score(outputs, Y))
    return loss, score

  def test(self, epoch, step):
    self.model.eval()

    loss = 0
    score = 0
    count = 0

    for X, Y in tqdm(self.testloader, desc="Evaluation", leave=False):
      b_loss, b_score = self.test_batch(X, Y)
      b_loss, b_score = float(b_loss), float(b_score)
      loss += b_loss
      score += b_score
      count += 1
    
    loss = loss / count
    score = score / count

    imgs = self.samples_to_wandb( self.model, self.device, self.device_cpu, self.samples)
    wandb.log({"epoch": epoch, "test_loss": loss, "score": score, "imgs": imgs}, step=step)
    self.model.train()

    return loss, score

  def train_batch(self, X, Y):
    X, Y = X.to(self.device), Y.to(self.device)
    outputs = self.model(X)
    loss = self.criterion(outputs, Y)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return float(loss)

  def samples_to_wandb(self, model, device, device_cpu, samples):
    if self.single_image:
      return [wandb.Image(TF.to_pil_image(model(x[None, :, :, :].to(device))[0].clamp(0.0, 1.0).to(device_cpu))) for x in samples]
    else:
      return [wandb.Image(TF.to_pil_image(model(x[None, :, :, :, :].to(device))[0].clamp(0.0, 1.0).to(device_cpu))) for x in samples]

  
  def train(self):

    train_loss = 0
    batch_count = 0
    elem_count = 0

    best_score = 0

    self.model.train()

    wandb.watch(self.model, log_freq=self.log_freq)
    
    with torch.autograd.set_detect_anomaly(True):
      for epoch in tqdm(range(self.epochs), desc="Epochs"):
        for X, Y in tqdm(self.trainloader, desc="Training", leave=False):
          batch_size = X.shape[0]

          loss_batch = self.train_batch(X, Y)

          # print(loss_batch)

          if self.freq_scheduler:
            self.scheduler.step()

          train_loss += loss_batch
          batch_count += 1
          elem_count += batch_size

          if batch_count % self.log_freq == 0:
            #print(train_loss/self.log_freq)
            lr = self.optimizer.param_groups[0]['lr']
            wandb.log({"train_loss": train_loss / self.log_freq, "learning_rate": lr}, step=elem_count)
            train_loss = 0

          if batch_count % self.eval_freq == 0:
            test_loss, score = self.test(epoch, elem_count)

            if score > best_score: 
              self.save_checkpoint(epoch, test_loss, score)
              best_score = score

        if not self.freq_scheduler:
          self.scheduler.step(test_loss)
      
      test_loss, score = self.test(epoch, elem_count)
      if score > best_score: 
        self.save_checkpoint(epoch, test_loss, score)
        best_score = score

  def save_checkpoint(self, epoch, loss, score):
    torch.save({
      'epoch': epoch,
      'model': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'scheduler': self.scheduler.state_dict(),
      'loss': loss,
      'best_score': score,
    }, self.save_path)

  def load_checkpoint(self, path):
    checkpoint = torch.load(path)
    self.model.load(checkpoint['model'])
    self.optimizer.load(checkpoint['optimizer'])
    self.scheduler.load(checkpoint['scheduler'])
    self.epoch = checkpoint['epoch']
    self.epoch = checkpoint['epoch']
    self.best_score = checkpoint['score']

