import torch.nn as nn
import torchvision.models as models

class StudentNetwork(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        base = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return self.fc(x)

