import torch
from MLOps_Project.data.fashion_mnist_dataset import get_dataloaders
import pdb

class TestData:
    def __init__(self) -> None:
        pass

    def setup_images_and_labels(self):
        print("setting up images and labels")
        train_loader, val_loader, test_loader = get_dataloaders()
        self.train_images, self.train_labels = train_loader.dataset.images, train_loader.dataset.labels
        self.val_images, self.val_labels = val_loader.dataset.images, val_loader.dataset.labels
        self.test_images, self.test_labels = test_loader.dataset.images, test_loader.dataset.labels
        
    def teardown_images_and_labels(self):
        print("tearing down images and labels")
        del self.train_images
        del self.train_labels
        del self.test_images
        del self.test_labels

    def test_image_label_shapes(self):
        self.setup_images_and_labels()
        # image tests
        assert self.train_images.shape[1:4] == torch.Size([1, 28, 28]), f'Image dimensionality mismatch! Expected dimensions (1 x 28 x 28) got {self.train_images.shape[1:4]}.'
        assert self.train_images.shape[1:4] == self.test_images.shape[1:4], 'Dimensionality of train and test images does not match!'
        # labels tests
        assert len(self.train_labels.shape) == 1, 'The labels in the train dataset are not 1D.'
        assert len(self.test_labels.shape) == 1, 'The labels in the train dataset are not 1D.'
        self.teardown_images_and_labels()
    
    def test_train_test_stratification(self):
        self.setup_images_and_labels()
        unique_labels = self.train_labels.unique()
        train_hist = torch.tensor([len(self.train_labels[self.train_labels == e])/len(self.train_labels) for e in unique_labels])
        test_hist = torch.tensor([len(self.test_labels[self.test_labels == e]) / len(self.test_labels) for e in unique_labels])
        train_test_dif = sum(train_hist - test_hist).item()
    

if __name__ == '__main__':
    TD = TestData()
    TD.test_train_test_stratification()

    