from models.baseline_model import Baseline_Model
import timm 
import torch


# Define class for VGG16 model using the TIMM framework
# DONT USE THIS MODEL - OUR IMAGES ARE TOO SMALL FOR THIS AMOUNT OF POOLING
class VGG(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="VGG16Model")
        self.lr = lr
        self.model = timm.create_model("vgg16", num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()
