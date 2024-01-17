from MLOps_Project.models.baseline_model import Baseline_Model
import torch


def predict_dataset(model: Baseline_Model, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch, _ in dataloader], 0)


def predict_batch(model: Baseline_Model, batch):  # Expected batch to be of shape [n, 3, 28, 28]
    model.eval()  # Make sure the model is in eval mode
    return model.forward(batch)


def predict_single(model: Baseline_Model, img):
    """Run Prediction for a given model and a single img of shape (3, 28, 28)"""
    img = img.unsqueeze(0)  # Change shape from (3, 28, 28) to (1, 3, 28, 28)
    model.eval()  # Make sure the model is in eval mode
    return model.forward(img)
