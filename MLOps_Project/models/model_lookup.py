from MLOps_Project.models.resnet import ResNet34
from MLOps_Project.models.efficientnet import EfficientNet
from MLOps_Project.models.mobilenet import MobileNet
from MLOps_Project.models.visiontransformer import VisionTransformer
from MLOps_Project.models.xcit import XcitNano
from MLOps_Project.models.baseline_model import Baseline_Model

### File for creating models based on their name


class ModelLookup:
    def find(name: str, optimizer_name, lr, dropout_rate, required_channels) -> Baseline_Model:
        name = name.lower()  # Lowercase to avoid case sensitive issues
        if name == "resnet":
            return ResNet34(optimizer_name, lr, dropout_rate, required_channels)
        elif name == "efficientnet":
            return EfficientNet(optimizer_name, lr, dropout_rate, required_channels)
        elif name == "mobilenet":
            return MobileNet(optimizer_name, lr, dropout_rate, required_channels)
        elif name == "visiontransformer":
            return VisionTransformer(optimizer_name, lr, dropout_rate, required_channels)
        elif name == "xcitnano":
            return XcitNano(optimizer_name, lr, dropout_rate, required_channels)
