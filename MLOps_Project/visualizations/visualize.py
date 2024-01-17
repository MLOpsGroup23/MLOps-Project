import torch
from models.model import ResNet34
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# This is a Visualization example
# Requirement: Run the training loop once, such that LightningTrainedModel.ckpt file is stored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get Test Data
# Download and load the test data

test_data = torch.load("./data/processed/train.pt")
test_dataset = TensorDataset(test_data[0], test_data[1])  # create your datset
testloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

dataiter = iter(testloader)
test_images, test_labels = next(dataiter)

test_images, test_labels = test_images.to(device), test_labels.to(device)

print("Loading model")
# Make Prediction and get Feature Maps using .forward_features method
model = ResNet34.load_from_checkpoint(checkpoint_path="./models/LightningTrainedModel2-v1.ckpt")
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
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=test_labels, cmap="tab10")

plt.colorbar(scatter)
plt.title("t-SNE Visualization of the Data")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.savefig("reports/figures/PredictionTSNE123.png")
