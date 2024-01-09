import torch
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import os

def get_FashionMNIST_dataset(cfg: DictConfig, train=True):
    print("Raw data directory: ", cfg.data.raw_dir)
    # Define a transform to normalize the data
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

    processed_data = (processed_images, processed_labels)
    # dimensions: torch.Size([60000, 1, 28, 28]) torch.Size([60000])

    # Save the processed dataset
    print("Saving train data to: ", filename)
    torch.save(processed_data, os.path.join(processed_dir, filename))

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    data_train = get_FashionMNIST_dataset(cfg, train=True)
    data_test = get_FashionMNIST_dataset(cfg, train=False)
    data_loader_train = DataLoader(data_train, batch_size=cfg.data.batch_size, shuffle=True)
    data_loader_test = DataLoader(data_test, batch_size=cfg.data.batch_size, shuffle=True)
    preprocess_FashionMNIST_dataset(cfg, data_loader_train, filename="train.pt")
    preprocess_FashionMNIST_dataset(cfg, data_loader_test, filename="test.pt")

if __name__ == '__main__':
    main()
    
