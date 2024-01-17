import torch
from torch.utils.data import DataLoader, TensorDataset
from MLOps_Project.predict_model import predict_dataset, predict_batch, predict_single
from MLOps_Project.models.resnet import ResNet34
from hydra import initialize, compose


class TestPredict:
    def test_predict_dataset(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            test_data = torch.load(cfg.data.processed_dir + "/test.pt")

            # Limiting the dataset to at most 100 elements
            subset_size = min(100, len(test_data[0]))
            test_data_subset = (test_data[0][:subset_size], test_data[1][:subset_size])

            test_dataset = TensorDataset(*test_data_subset)  # create your dataset with subset
            testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

            model = ResNet34(lr=0.003, dropout_rate=0.2, required_channels=3)  # Random untrained model
            predict_dataset(model, testloader)

    def test_predict_batch(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            test_data = torch.load(cfg.data.processed_dir + "/test.pt")
            test_dataset = TensorDataset(test_data[0], test_data[1])  # create your dataset
            testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
            dataiter = iter(testloader)
            test_images, _ = next(dataiter)

            model = ResNet34(lr=0.003, dropout_rate=0.2, required_channels=3)  # Random untrained model
            predict_batch(model, test_images)

    def test_predict_single(self):
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="config")
            test_data = torch.load(cfg.data.processed_dir + "/test.pt")
            test_dataset = TensorDataset(test_data[0], test_data[1])  # create your dataset
            testloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
            dataiter = iter(testloader)
            test_images, _ = next(dataiter)

            model = ResNet34(lr=0.003, dropout_rate=0.2, required_channels=3)  # Random untrained model
            predict_single(model, test_images[0])
