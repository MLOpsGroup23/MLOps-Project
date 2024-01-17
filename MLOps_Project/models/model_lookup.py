from MLOps_Project.models.resnet import ResNet34
from MLOps_Project.models.efficientnet import EfficientNet
from MLOps_Project.models.mobilenet import MobileNet
from MLOps_Project.models.visiontransformer import VisionTransformer
from MLOps_Project.models.xcit import XcitNano
from MLOps_Project.models.baseline_model import Baseline_Model

### File for creating models based on their name


class ModelLookup:
    def find(name: str, lr, dropout_rate, required_channels) -> Baseline_Model:
        if name.lower() == "resnet":
            return ResNet34(name, lr, dropout_rate, required_channels)
        elif name.lower() == "efficientnet":
            return EfficientNet(name, lr, dropout_rate, required_channels)
        elif name.lower() == "mobilenet":
            return MobileNet(name, lr, dropout_rate, required_channels)
        elif name.lower() == "visiontransformer":
            return VisionTransformer(name, lr, dropout_rate, required_channels)
        elif name.lower() == "xcitnano":
            return XcitNano(name, lr, dropout_rate, required_channels)
