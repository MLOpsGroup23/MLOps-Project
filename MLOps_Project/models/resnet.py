from MLOps_Project.models.baseline_model import Baseline_Model
import timm 
import torch

# Define class for ResNet34 model using the TIMM framework
class ResNet34(Baseline_Model):
    def __init__(self, optimizer_name='Adam', lr=0.003, dropout_rate=0.2, required_channels=3):
        super().__init__(filename="ResNetModel")
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.model = timm.create_model('resnet34', num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()

        # Setup first convolutional layer to work with a single channel input
        # Removed as we want to train on 3 channel input
        # old_conv_layer = self.model.conv1
        # self.model.conv1 = torch.nn.Conv2d(1, old_conv_layer.out_channels, 
        #                       kernel_size=old_conv_layer.kernel_size, 
        #                       stride=old_conv_layer.stride, 
        #                       padding=old_conv_layer.padding, 
        #                       bias=old_conv_layer.bias)
        # if old_conv_layer.in_channels == 3:
        #     with torch.no_grad():
        #         self.model.conv1.weight[:,:] = old_conv_layer.weight[:,:].mean(dim=1, keepdim=True)