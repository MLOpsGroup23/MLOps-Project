from MLOps_Project.models.baseline_model import Baseline_Model
import timm 
import torch
from torchvision import transforms

# Define class for a VisionTransformer model using the TIMM framework
class VisionTransformer(Baseline_Model):
    def __init__(self, lr=0.003, dropout_rate=0.2):
        super().__init__()
        self.lr = lr
        self.model = timm.create_model('vit_small_patch16_18x2_224', img_size=28, num_classes=10, drop_rate=dropout_rate)
        self.loss = torch.nn.CrossEntropyLoss()

        # Setup first convolutional layer to work with a single channel input
        # old_conv_layer = self.model.conv1
        # self.model.conv1 = torch.nn.Conv2d(1, old_conv_layer.out_channels, 
        #                       kernel_size=old_conv_layer.kernel_size, 
        #                       stride=old_conv_layer.stride, 
        #                       padding=old_conv_layer.padding, 
        #                       bias=old_conv_layer.bias)
        # if old_conv_layer.in_channels == 3:
        #     with torch.no_grad():
        #         self.model.conv1.weight[:,:] = old_conv_layer.weight[:,:].mean(dim=1, keepdim=True)
    
    
    def forward(self, x):
        x = x.repeat_interleave(3, dim=1) # Resize input channels to 3 channels
        return self.model(x)