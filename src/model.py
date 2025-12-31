import torch.nn as nn
from torchvision import models
from .config import DEVICE

def build_model(num_classes):
    print("🏗️ Building EfficientNetV2-Medium...")
    model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4), 
        nn.Linear(in_features, num_classes)
    )
    model = model.to(DEVICE)
    return model
