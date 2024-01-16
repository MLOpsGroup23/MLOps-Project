from MLOps_Project.models.resnet import ResNet34 
import pytest
import torch

class TestModel:

    def setup_resnet_model(self):
        self.model = ResNet34()
    
    def teardown_resnet_model(self):
        del self.model

    def test_forward_resnet(self):
        self.setup_resnet_model()
        randomData = torch.rand((10, 3, 28, 28))
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"
        self.teardown_resnet_model()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Unable to test CUDA forward id cuda device is not available")
    def test_forward_resnet_cuda(self):
        self.setup_resnet_model()
        self.model.to('cuda')
        randomData = torch.rand((10, 3, 28, 28)).to('cuda')
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"
        self.teardown_resnet_model()
    
    def test_something(self):
        pass
    

    