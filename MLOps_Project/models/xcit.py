from MLOps_Project.models.baseline_model import Baseline_Model
import timm 
import torch

# Define class for XCIT Nano model using the TIMM framework
class XcitNano(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="xcitModel")
        self.lr = lr
        self.model = timm.create_model('xcit_nano_12_p16_224', num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()