import torch
import argparse
from pytorch_lightning import Trainer
from models.model import ResNet34
from torch.utils.data import DataLoader, TensorDataset



def main(save_location="./", n_epochs=1):
    # Create Data Loaders and Load Data Sets
    train_data = torch.load("./data/processed/train.pt")
    test_data = torch.load("./data/processed/test.pt")

    train_dataset = TensorDataset(train_data[0], train_data[1])
    test_dataset  = TensorDataset(test_data[0], test_data[1])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False) 

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
    