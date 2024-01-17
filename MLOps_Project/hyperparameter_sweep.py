from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from MLOps_Project.data.fashion_mnist_dataset import get_dataloaders
from MLOps_Project.models.resnet import ResNet34
import hydra
from omegaconf import OmegaConf
import wandb

# TODO: (Could have) Enum class for enabling sweep with different models

def train(config=None):
    """
    Define wrapper function for training compatible with wandb sweep agent
    """
    with wandb.init(config=config):
        wandb_logger = WandbLogger(project='FashionMNIST')
        config=wandb.config
        # Initialize a new wandb run
        # train_dataloader, val_dataloader, _ = get_dataloaders_for_sweep()
        model = ResNet34(
                    optimizer_name=config['optimizer_name'],
                    lr=config['lr'], 
                    dropout_rate=config['dropout_rate'])
    
        trainer = Trainer(callbacks=model.callbacks, max_epochs=config['n_epochs'], logger=wandb_logger)
        trainer.fit(model, train_dataloader, val_dataloader)



if __name__ == '__main__':
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name="config")
    sweep_config = OmegaConf.to_container(config.sweep, resolve=True)
    
    train_dataloader, val_dataloader, _ = get_dataloaders(config)
    # hyperparameter_sweep(config=sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project='FashionMNIST')
    wandb.agent(sweep_id, train, count=10)