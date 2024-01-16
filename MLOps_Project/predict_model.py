from MLOps_Project.models.resnet import ResNet34
import torch

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

def predict_single(model: ResNet34, img):
    """ Run Prediction for a given model and a single img of shape (1, 28, 28)
    """
    img = img.unsqueeze(0) # Change shape from (1, 28, 28) to (1, 1, 28, 28)
    model.eval() # Make sure the model is in eval mode
    return model.forward(img)
