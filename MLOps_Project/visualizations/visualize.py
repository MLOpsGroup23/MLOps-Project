import torch
from pytorch_lightning import Trainer
from MLOps_Project.models.model import ResNet34
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# This is a Visualization example
# Requirement: Run the training loop once, such that LightningTrainedModel.ckpt file is stored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get Test Data
# Download and load the test data 
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=True)
dataiter = iter(testloader)
test_images, test_labels = next(dataiter)

test_images, test_labels = test_images.to(device), test_labels.to(device)

print("Loading model")
# Make Prediction and get Feature Maps using .forward_features method
model = ResNet34.load_from_checkpoint(checkpoint_path="./models/LightningTrainedModel.pt-v11.ckpt")
model = model.to(device)

print("Making Prediction")
cnnOut = model.model.forward_features(test_images)

print("Reshaping Data")
data_reshaped = cnnOut.reshape(cnnOut.shape[0], -1).detach().numpy()

print("Applying t-SNE")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
data_2d = tsne.fit_transform(data_reshaped)
print(data_2d.shape)

# Plotting
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=test_labels, cmap='tab10')

plt.colorbar(scatter)
plt.title("t-SNE Visualization of the Data")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.savefig("reports/figures/PredictionTSNE.png")