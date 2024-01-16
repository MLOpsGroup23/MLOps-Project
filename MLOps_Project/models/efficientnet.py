from MLOps_Project.models.baseline_model import Baseline_Model
import timm
import torch


# Define class for EfficientNet model using the TIMM framework
class EfficientNet(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="EfficientNetModel")
        self.lr = lr
        self.model = timm.create_model("efficientnet_b1_pruned", num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()
