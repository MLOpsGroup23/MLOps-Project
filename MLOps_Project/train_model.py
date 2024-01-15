import torch
import argparse
from pytorch_lightning import Trainer
from models.model import ResNet34
from torch.utils.data import DataLoader, TensorDataset
from data.fashion_mnist_dataset import get_dataloaders



def main(save_location="./", n_epochs=1):
    # Create custom datasets and dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

    # Test Training
    model = ResNet34()

    trainer = Trainer(callbacks=model.callbacks, max_epochs=n_epochs, default_root_dir=save_location)
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training.')
    parser.add_argument('--save_location', type=str, help='Location to save the trained model.')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs to train for.')

    args = parser.parse_args()
    print(args.save_location)
    print(args.n_epochs)
    main(save_location=args.save_location, n_epochs=args.n_epochs)
    