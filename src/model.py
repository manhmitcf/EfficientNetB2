import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B2_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=3):  # Truyền num_classes vào __init__
        super(FishClassifier, self).__init__()
        # Load EfficientNetB2 pre-trained
        self.efficientnet = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        # Thay đổi fully connected layer cuối cùng để phù hợp với số lớp mới
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)