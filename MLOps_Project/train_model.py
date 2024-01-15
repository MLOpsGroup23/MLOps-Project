import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from MLOps_Project.models.model import ResNet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from data.fashion_mnist_dataset import get_dataloaders

# Initialize logger
wandb_logger = WandbLogger(project='FashionMNIST')

# Create custom datasets and dataloaders
train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

# Test Training
model = ResNet34()

trainer = Trainer(callbacks=model.callbacks, max_epochs=5, logger=wandb_logger)
trainer.fit(model, train_dataloader, val_dataloader)

print("!!! DONE !!!")


