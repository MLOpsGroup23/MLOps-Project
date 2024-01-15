from MLOps_Project.models.resnet import ResNet34 
import torch

def test_train():
    randomData = torch.rand((10, 1, 28, 28))
    model = ResNet34()
    pred = model.forward(randomData)

    assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"