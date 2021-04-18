import data
import trainer
import models
import wandb
import device

def run_experiment(cfg_dict):

  device = utils.get_device()

  wandb.login()

  with wandb.init(project=cfg_dict['project_name'], config=cfg_dict, notes=cfg_dict['run_description']) as wandb_run:
    cfg = wandb_run.config

    model = models.make_model(cfg.model).to(device)
    trainset = data.make_dataset(**cfg.train_dataset.dataset)
    testset = data.make_dataset(**cfg.test_dataset.dataset)
    sample = [testset[i][0] for i in range(8)]

    trainloader = data.make_loader(trainset, **cfg.train_dataset.dataloader)
    testloader = data.make_loader(testset, **cfg.test_dataset.dataloader)

    save_path = cfg.save_dir + '/' + wandb_run.name + "_best.pt"
    trainer = trainer.Trainer(save_path, device, model, trainloader, testloader, samples, **cfg.trainer)

    trainer.train()


def main():
  cfg_dict = utils.load_config()
  run_experiment(cfg_dict)


if __name__=="__main__":
  main()




