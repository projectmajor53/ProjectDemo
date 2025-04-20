import torch.nn as nn

class KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, student, teacher):
        return self.criterion(student, teacher)

