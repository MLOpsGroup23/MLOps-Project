import torch
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_FashionMNIST_dataset(cfg: DictConfig, train=True):
    print("Raw data directory: ", cfg.data.raw_dir)
    dat = datasets.FashionMNIST(root=cfg.data.raw_dir, train=train, download=True, transform=transforms.ToTensor())
    return dat

def preprocess_FashionMNIST_dataset(cfg: DictConfig, data: DataLoader, filename: str):
    processed_dir = cfg.data.processed_dir
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    processed_images = None
    processed_labels = None

    # Process and store the dataset
    for batch_idx, (images, labels) in enumerate(data):
        processed_images = images if batch_idx == 0 else torch.cat((processed_images, images), dim=0)
        processed_labels = labels if batch_idx == 0 else torch.cat((processed_labels, labels), dim=0)

    processed_images = (processed_images - torch.mean(processed_images))/torch.std(processed_images)

    # If config requires data to have more than 1 channels, add extra channels
    if(cfg.data.channels > 1):
        processed_images = processed_images.repeat_interleave(cfg.data.channels, dim=1)

    processed_data = (processed_images, processed_labels)
    # dimensions: torch.Size([60000, channels, 28, 28]) torch.Size([60000])

    # Save the processed dataset
    print("Saving train data to: ", filename)
    torch.save(processed_data, os.path.join(processed_dir, filename))

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Load the FashionMNIST test dataset
    data_test = get_FashionMNIST_dataset(cfg, train=False)
    # Load the full FashionMNIST training dataset
    full_train_dataset = get_FashionMNIST_dataset(cfg, train=True)

    # Get the targets from the dataset for class distribution analysis
    targets = np.array(full_train_dataset.targets)

    # Split indices while maintaining class distribution
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=cfg.data.val_size,
        stratify=targets,
        random_state=cfg.data.seed  # for reproducibility
    )

    # Create subsets for train and validation
    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    # Create data loaders
    data_loader_train = DataLoader(train_subset, batch_size=cfg.data.batch_size, shuffle=True)
    data_loader_val = DataLoader(val_subset, batch_size=cfg.data.batch_size, shuffle=False)
    data_loader_test = DataLoader(data_test, batch_size=cfg.data.batch_size, shuffle=True)
    preprocess_FashionMNIST_dataset(cfg, data_loader_train, filename="train.pt")
    preprocess_FashionMNIST_dataset(cfg, data_loader_val, filename="val.pt")
    preprocess_FashionMNIST_dataset(cfg, data_loader_test, filename="test.pt")

if __name__ == '__main__':
    main()
    
