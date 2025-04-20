import torch
import torch.nn as nn

class OcclusionAdapter(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.w_p = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.w_s = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.w_a = nn.Parameter(torch.randn(feature_dim, feature_dim))

    def forward(self, f_s, w_t):
        f_a = f_s * (w_t @ self.w_p)  # point-wise fusion
        dynamic_weight = torch.sigmoid(f_a @ self.w_a)
        weighted_f_s = torch.sigmoid(f_s @ self.w_s)
        return f_a + dynamic_weight * weighted_f_s
