from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import hydra
from omegaconf import DictConfig

def plt_savefig(fig, fig_name, dir="./reports/figures/"):
    # save both as pdf and png
    path = dir + fig_name + ".pdf"
    fig.savefig(path, bbox_inches='tight', pad_inches=0.2, format='pdf', dpi=300)
    print("Saved to: ", path)
    path = dir + fig_name + ".png"
    fig.savefig(path, bbox_inches='tight', pad_inches=0.2, format='png', dpi=300)
    print("Saved to: ", path)

def plot_images(dataset: TensorDataset):
    # plot 5 by 5
    fig, axs = plt.subplots(4, 5, figsize=(14, 10))

    rand_idx = torch.randperm(len(dataset))[:20]

    # plot 3 images from each class
    for i in range(4):
        for j in range(5):
            axs[i, j].imshow(dataset[rand_idx[i*5+j]][0].reshape(28, 28), cmap="gray")
            # Set class label as title
            axs[i, j].set_title(f"Class: {dataset[i*5+j][1].item()}")
            axs[i, j].axis('off')

    plt.suptitle("Fashion MNIST Dataset (random samples)")
    plt.tight_layout()

    return plt

def plot_class_distirbution(dataset, title="Fashion MNIST Dataset Class Distribution"):
    # count number of samples for each class
    counts = torch.zeros(10)
    for _, label in dataset:
        counts[label] += 1

    # plot class distribution
    fig, ax = plt.subplots()
    ax.bar(range(10), counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of samples")
    ax.set_title(title)
    plt.xticks(range(10), [str(i) for i in range(10)])
    plt.tight_layout()

    return plt

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Create Data Loaders and Load Data Sets

    train_data = torch.load(cfg.data.processed_dir + "/train.pt")

    dataset = TensorDataset(train_data[0], train_data[1])  # create your datset


    plot = plot_images(dataset)
    plt_savefig(plot, "FashionMNIST_Dataset_samples")

    plot = plot_class_distirbution(dataset, title="Fashion MNIST Dataset Class Distribution (train)")
    plt_savefig(plot, "FashionMNIST_Dataset_class_distribution")

    test_data = torch.load(cfg.data.processed_dir + "/test.pt")
    test_dataset = TensorDataset(test_data[0], test_data[1])  # create your datset

    plot = plot_class_distirbution(dataset, title="Fashion MNIST Dataset Class Distribution (test)")
    plt_savefig(plot, "FashionMNIST_Dataset_class_distribution_test")

    val_data = torch.load(cfg.data.processed_dir + "/val.pt")
    val_dataset = TensorDataset(val_data[0], val_data[1])  # create your datset

    plot = plot_class_distirbution(dataset, title="Fashion MNIST Dataset Class Distribution (validation)")
    plt_savefig(plot, "FashionMNIST_Dataset_class_distribution_val")

if __name__ == '__main__':
    main()
