from MLOps_Project.models.resnet import ResNet34 
import pytest
import torch

class TestModel:

    def setup_model(self):
        self.model = ResNet34()
    
    def teardown_model(self):
        del self.model

    def test_forward(self):
        self.setup_model()
        randomData = torch.rand((10, 1, 28, 28))
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"
        self.teardown_model()
    
    @pytest.mark.skipif(not torch.cuda.is_available())
    def test_forward_cuda(self):
        self.setup_model()
        self.model.to('cuda')
        randomData = torch.rand((10, 1, 28, 28)).to('cuda')
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"
        self.teardown_model()
    
    def test_something(self):
        pass
    

    