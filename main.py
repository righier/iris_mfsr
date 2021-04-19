import argparse

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

def run_experiment(cfg_dict):

  device = utils.get_device()

  wandb.login()

  with wandb.init(project=cfg_dict['project_name'], config=cfg_dict, notes=cfg_dict['run_description']) as wandb_run:
    cfg = wandb_run.config

    if cfg.model['name'] == '2dsrnet':
      cfg.train_dataset['dataset']['single_image'] = True
      cfg.test_dataset['dataset']['single_image'] = True
    else:
      cfg.train_dataset['dataset']['single_image'] = False
      cfg.test_dataset['dataset']['single_image'] = False

    model = models.make_model(**cfg.model).to(device)
    model = model.apply(models.init_weights)

    trainloader = data.make_loader(**cfg.train_dataset)
    testloader = data.make_loader(**cfg.test_dataset)
    samples = [testloader.dataset[i][0] for i in range(8)]

    save_path = cfg.save_dir + '/' + wandb_run.name + "_best.pt"
    train = trainer.Trainer(save_path, device, model, trainloader, testloader, samples, **cfg.trainer)

    train.train()

  


def main(args):
  cfg_dict = utils.load_config(args.cfg)
  run_experiment(cfg_dict)


if __name__=="__main__":
  args = parser()
  main(args)




