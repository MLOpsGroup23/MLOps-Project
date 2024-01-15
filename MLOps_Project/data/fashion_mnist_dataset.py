import torch
import hydra
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import RandomRotation, Compose
from omegaconf import DictConfig
import pdb

class FashionMNISTDataset(Dataset):
    def __init__(self, cfg, data, transformations = None) -> None:
        super().__init__()
        self.images = data[0] # N x channels x H x W
        self.labels = data[1] # N
        self.transformations = transformations

        if(cfg.architecture.required_channels != self.images.shape[1]):
            raise BaseException("The model used requires " + str(cfg.architecture.required_channels) + " channels, but loaded data has " + str(self.images.shape[1]) + " channels.")

    def __getitem__(self, index) -> any:
        image = self.images[index]
        if self.transformations:
            image = self.transformations(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

def compose_transformations(cfg):
    if cfg is None:
        return
    transforms = []
    options = {'rotation' : RandomRotation}
    for transform in cfg.keys():
        if transform in options.keys():
            transforms.append(options[transform](**cfg[transform]))
    return Compose(transforms)
    



def get_dataset(cfg, split: str = 'train'):
    shuffle = True if split == 'train' else False
    data = torch.load(os.path.join(cfg.data.processed_dir, f'{split}.pt'))
    transformations = compose_transformations(cfg['data'][split]['transformations'])
    dataset = FashionMNISTDataset(cfg, data, transformations)
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle, num_workers=cfg.data.num_workers)
    return dataloader

def get_dataloaders(cfg: DictConfig):
    train_loader = get_dataset(cfg=cfg, split='train')
    val_loader = get_dataset(cfg=cfg, split='val')
    test_loader = get_dataset(cfg=cfg, split='test')
    
    return train_loader, val_loader, test_loader