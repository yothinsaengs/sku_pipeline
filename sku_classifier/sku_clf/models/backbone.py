import torch
import torch.nn as nn
import timm
from typing import Dict, Any

class SKUClassifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(SKUClassifier, self).__init__()
        backbone_name = config['model']['backbone']
        pretrained = config['model'].get('pretrained', True)
        num_classes = config['model']['num_classes']
        dropout_p = config['model']['head'].get('dropout', 0.3)
        global_pool = config['model']['head'].get('global_pool', 'avg')

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=''  # Remove global pool to use our own
        )

        # Get number of features from backbone
        # We can do this by running a dummy forward pass or checking model attribute
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # Head
        if global_pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif global_pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported global_pool: {global_pool}")

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.num_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim == 4:
            x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def get_model(config: Dict[str, Any]) -> nn.Module:
    return SKUClassifier(config)
