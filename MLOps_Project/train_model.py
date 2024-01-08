import torch
from pytorch_lightning import Trainer
from MLOps_Project.models.model import ResNet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset



# Create Data Loaders and Load Data Sets
train_data = torch.load("./data/processed/test.pt")
test_data = torch.load("./data/processed/train.pt")

train_dataset = TensorDataset(train_data[0], train_data[1])  # create your datset
test_dataset  = TensorDataset(test_data[0], test_data[1])  # create your datset

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # create your dataloader
val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)  # create your dataloader

# Test Training
model = ResNet34()

trainer = Trainer(callbacks=model.callbacks, max_epochs=5)
trainer.fit(model, train_dataloader, val_dataloader)

print("!!! DONE !!!")


