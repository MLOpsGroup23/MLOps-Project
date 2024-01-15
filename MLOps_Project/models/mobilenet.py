from models.baseline_model import Baseline_Model
import timm 
import torch

# Define class for MobileNet model using the TIMM framework
class MobileNet(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="MobileNetModel")
        self.lr = lr
        self.model = timm.create_model('mobilenetv3_small_050', num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()