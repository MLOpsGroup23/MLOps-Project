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
        pdb.set_trace()
        self.transformations = Compose(self.__get_transformations())
        
    def __get_transformations(self):

        rotate = RandomRotation(degrees=self.config.transformations.rotation.degrees,
                                interpolation=InterpolationMode.BILINEAR)
        
        scale = None
        transforms = {'rotation': rotate}
        return [transform for name, transform in transforms.items() if self.config['transformations'][f'{name}']['use']]

    def __getitem__(self, index) -> any:
        image = self.transformations(self.images[index])
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.labels)


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def get_train_dataset(cfg : DictConfig):
    data = torch.load(os.path.join(cfg.data.processed_dir, 'train.pt'))
    train_dataset = FashionMNISTDataset(data, cfg.data.train)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def get_test_dataset(cfg):
    data = torch.load(os.path.join(cfg.data.processed_dir, 'test.pt'))
    test_dataset = hydra.utils.instantiate(cfg.data.test)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)


@hydra.main(version_base=None, config_path='../../configs', config_name='config')
def get_val_dataset(cfg):
    data = torch.load(os.path.join(cfg.data.processed_dir, 'val.pt'))
    val_dataset = hydra.utils.instantiate(cfg.data.test)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)

if __name__ == '__main__':
    get_train_dataset()
