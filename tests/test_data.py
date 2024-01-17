import torch
from hydra import initialize, compose
from MLOps_Project.data.fashion_mnist_dataset import get_dataloaders
from MLOps_Project.data.create_bmp_img import create_random_fashion_img
from MLOps_Project.data.make_dataset import make_datasets
from os.path import exists


class TestData:
    def setup_images_and_labels(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            print("setting up images and labels")
            train_loader, val_loader, test_loader = get_dataloaders(cfg)
            self.train_images, self.train_labels = train_loader.dataset.images, train_loader.dataset.labels
            self.val_images, self.val_labels = val_loader.dataset.images, val_loader.dataset.labels
            self.test_images, self.test_labels = test_loader.dataset.images, test_loader.dataset.labels

    def teardown_images_and_labels(self):
        print("tearing down images and labels")
        del self.train_images
        del self.train_labels
        del self.test_images
        del self.test_labels
        del self.val_images
        del self.val_labels

    def test_image_label_shapes(self):
        self.setup_images_and_labels()
        # image tests
        assert self.train_images.shape[1:4] == torch.Size(
            [3, 28, 28]
        ), f"Image dimensionality mismatch! Expected dimensions (3 x 28 x 28) got {self.train_images.shape[1:4]}."
        assert (
            self.train_images.shape[1:4] == self.test_images.shape[1:4]
        ), "Dimensionality of train and test images does not match!"
        assert (
            self.train_images.shape[1:4] == self.val_images.shape[1:4]
        ), "Dimensionality of train and validation images does not match!"
        # labels tests
        assert len(self.train_labels.shape) == 1, "The labels in the train dataset are not 1D."
        assert len(self.test_labels.shape) == 1, "The labels in the test dataset are not 1D."
        assert len(self.val_labels.shape) == 1, "The labels in the validation dataset are not 1D."
        self.teardown_images_and_labels()

    def test_stratification_criteria(self):
        TOL = 0.01
        self.setup_images_and_labels()
        unique_labels = self.train_labels.unique()
        train_hist = torch.tensor(
            [len(self.train_labels[self.train_labels == e]) / len(self.train_labels) for e in unique_labels]
        )
        test_hist = torch.tensor(
            [len(self.test_labels[self.test_labels == e]) / len(self.test_labels) for e in unique_labels]
        )
        val_hist = torch.tensor(
            [len(self.val_labels[self.val_labels == e]) / len(self.val_labels) for e in unique_labels]
        )
        train_test_dif = sum(abs(train_hist - test_hist)).item()
        train_val_dif = sum(abs(train_hist - val_hist)).item()
        assert (
            train_test_dif < TOL
        ), "The difference in class imbalances for train and test exceeds the desired threshold"
        assert (
            train_val_dif < TOL
        ), "The difference in class imbalances for train and test exceeds the desired threshold"
        self.teardown_images_and_labels()

    def test_create_bmp_image(self):
        create_random_fashion_img("./data/processed/test.pt", "testingImage", "./outputs")
        assert exists(
            "./outputs/testingImage.bmp"
        ), "When creating image, it was expected that a file was created, but this was not the case"

    def test_make_dataset(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            make_datasets(cfg)

            assert exists("./data/processed/train.pt"), "Expected the training set to be made"
            assert exists("./data/processed/val.pt"), "Expected the validation set to be made"
            assert exists("./data/processed/test.pt"), "Expected the testing set to be made"
