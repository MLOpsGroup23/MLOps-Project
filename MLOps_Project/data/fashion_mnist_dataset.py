import torch
import hydra
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import RandomRotation, InterpolationMode, Compose
from omegaconf import OmegaConf, DictConfig
import pdb

class FashionMNISTDataset(Dataset):
    def __init__(self, data, config) -> None:
        super().__init__()
        print(config)
        # data = torch.load(data_path)
        self.config = config
        self.images = data[0] # N x 1 x H x W
        self.labels = data[1] # N
        self.__get_transformations()
        
    def __get_transformations(self):

        rotate = RandomRotation(degrees=self.config.transformations.rotation.degrees,
                                interpolation=InterpolationMode.BILINEAR)
        scale = None
        transforms = {'rotation': rotate}

        augmentations = [transform for name, transform in transforms.items() if self.config['transformations'][f'{name}']['use']]
        if len(augmentations):
            self.transformations = Compose(augmentations)
        else:
            self.transformations = None


    def __getitem__(self, index) -> any:
        image = self.transformations(self.images[index]) if self.transformations else self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


def get_dataset(cfg, split: str = 'train'):
    shuffle = True if split == 'train' else False
    data = torch.load(os.path.join(cfg.data.processed_dir, f'{split}.pt'))
    dataset = FashionMNISTDataset(data, cfg['data'][split])
    dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle)
    return dataloader

@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    train_loader = get_dataset(cfg, split='train')
    _x = next(iter(train_loader))
    # pdb.set_trace()
    # val_loader = get_dataset(cfg, split='val')
    test_loader = get_dataset(cfg, split='test')
    _x = next(iter(test_loader))
    return train_loader, test_loader

if __name__ == '__main__':
    main()
