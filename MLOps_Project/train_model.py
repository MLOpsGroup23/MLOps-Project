import torch
from pytorch_lightning import Trainer
from MLOps_Project.models.model import ResNet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from data.fashion_mnist_dataset import get_train_test_dataloaders

# Create custom datasets and dataloaders
train_dataloader, val_dataloader = get_train_test_dataloaders()

# Test Training
model = ResNet34()

trainer = Trainer(callbacks=model.callbacks, max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)

print("!!! DONE !!!")


