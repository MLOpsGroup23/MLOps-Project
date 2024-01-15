from MLOps_Project.models.baseline_model import Baseline_Model
import timm 
import torch
from torchvision import transforms

# Define class for a VisionTransformer model using the TIMM framework
class VisionTransformer(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="VistionTransformsSmallModel")
        self.lr = lr
        self.model = timm.create_model('vit_small_patch16_18x2_224', img_size=28, num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()