from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from data.fashion_mnist_dataset import get_dataloaders
import hydra
from omegaconf import DictConfig
import wandb

# Call when training!
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Initialize logger
    wandb_logger = WandbLogger(project='FashionMNIST')

    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)

    print(cfg.architecture)

    model = hydra.utils.instantiate(cfg.architecture)
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.training.pl_basepath,
            monitor="val/accuracy",
            mode="max",
            filename=model.filename,
            save_on_train_epoch_end=True,
        )
    ]

    trainer = Trainer(callbacks=callbacks, max_epochs=cfg.training.max_epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    print("!!! DONE !!!")


# Entrypoint
if __name__ == '__main__':
    train()