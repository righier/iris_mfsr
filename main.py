import argparse
import os
import datetime

import data
import trainer
import models
import wandb
import utils

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config.json', type=str)
    args = parser.parse_args()
    return args

def expand_cfg(cfg):
  dataset_name = cfg['train_dataset']['dataset']['name']
  if dataset_name == 'mlfdb':
    mean = 0.3780295949066102
    std = 0.21023042003545392
  elif dataset_name in ('vimeo', 'vimeo_patches'):
    mean = 0.34557581469488063
    std = 0.25942671746370527 
  else:
    mean = 0.0
    std = 1.0

  cfg['model']['mean'] = mean
  cfg['model']['std'] = std

  if cfg['model']['name'] in ('2dsrnet', '2dwdsrnet'):
    cfg['train_dataset']['dataset']['single_image'] = True
    cfg['test_dataset']['dataset']['single_image'] = True
    cfg['trainer']['single_image'] = True
  else:
    cfg['train_dataset']['dataset']['single_image'] = False
    cfg['test_dataset']['dataset']['single_image'] = False
    cfg['trainer']['single_image'] = False
    
  scale = cfg['model']['scale']
  cfg['train_dataset']['dataset']['scale'] = scale
  cfg['test_dataset']['dataset']['scale'] = scale
  cfg['train_dataset']['dataset']['train'] = True
  cfg['test_dataset']['dataset']['train'] = False

def run_experiment(cfg_dict):

  device = utils.get_device()

  expand_cfg(cfg_dict)

  wandb.login()

  with wandb.init(project=cfg_dict['project_name'], config=cfg_dict, notes=cfg_dict['run_description']) as wandb_run:
    cfg = wandb_run.config

    model = models.make_model(**cfg.model).to(device)
    model = model.apply(models.init_weights)

    trainloader = data.make_loader(**cfg.train_dataset)
    testloader = data.make_loader(**cfg.test_dataset)
    samples = [testloader.dataset[i][0] for i in range(8)]


    if wandb_run.name:
      filename = wandb_run.name
    else:
      filename = "checkpoint_" + datetime.date.today().strftime("%d%m%Y")
    save_path = os.path.join(cfg.save_dir, filename + "_best.pt")
    train = trainer.Trainer(save_path, device, model, trainloader, testloader, samples, **cfg.trainer)

    train.train()

  


def main(args):
  cfg_dict = utils.load_config(args.cfg)
  run_experiment(cfg_dict)


if __name__=="__main__":
  args = parser()
  main(args)




