from MLOps_Project.models.model_lookup import ModelLookup
import pytest
import torch


class TestModel:
    def setup_model(self, name):
        self.model = ModelLookup.find(name, optimizer_name="Adam", lr=0.003, dropout_rate=0.2, required_channels=3)

    def teardown_model(self):
        del self.model

    def specific_model_testing(self, name):
        self.setup_model(name)
        randomData = torch.rand((10, 3, 28, 28))
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), (
            "Shape of the output from model " + name + " of the prediction should be [batchSize, 10]"
        )
        self.teardown_model()

    def test_forward_resnet(self):
        self.specific_model_testing("Resnet")

    def test_forward_efficientnet(self):
        self.specific_model_testing("EfficientNet")

    def test_forward_mobilenet(self):
        self.specific_model_testing("MobileNet")

    def test_forward_visiontransformer(self):
        self.specific_model_testing("VisionTransformer")

    def test_forward_xcit(self):
        self.specific_model_testing("XcitNano")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Unable to test CUDA forward id cuda device is not available"
    )
    def test_forward_resnet_cuda(self):
        self.setup_model("Resnet")
        self.model.to("cuda")
        randomData = torch.rand((10, 3, 28, 28)).to("cuda")
        pred = self.model.forward(randomData)
        assert pred.shape == torch.Size((10, 10)), "Shape of the output of the prediction should be [batchSize, 10]"
        self.teardown_resnet_model()
