import torch
from pytorch_lightning import Trainer
from MLOps_Project.models.model import ResNet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=len(testset), shuffle=True)

# Test Training
model = ResNet34()

trainer = Trainer(callbacks=model.callbacks, max_epochs=1)
trainer.fit(model, trainloader, testloader)

print("!!! DONE !!!")


