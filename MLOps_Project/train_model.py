from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from MLOps_Project.data.fashion_mnist_dataset import get_dataloaders
from MLOps_Project.models.model_lookup import ModelLookup
from hydra import initialize, compose
from omegaconf import DictConfig


# Call when training!
def train(cfg: DictConfig, wandb_offline=False):
    # Initialize logger and save experiment hyperparameters
    wandb_logger = WandbLogger(project="FashionMNIST", offline=wandb_offline)
    wandb_logger.experiment.config.update(dict(cfg.architecture))

    # Create custom datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(cfg)

    # Test Training
    model = ModelLookup.find(
        cfg.architecture.name,
        cfg.architecture.optimizer_name,
        cfg.architecture.lr,
        cfg.architecture.dropout_rate,
        cfg.architecture.required_channels,
    )
    print(model)

    trainer = Trainer(callbacks=model.callbacks, max_epochs=cfg.training.max_epochs, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    print("!!! DONE !!!")


# Entrypoint
if __name__ == "__main__":
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
        train(cfg, wandb_offline=False)
