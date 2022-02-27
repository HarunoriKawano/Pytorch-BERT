import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer normalization for BERT.
    """
    def __init__(self, feature_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_size))
        self.bias = nn.Parameter(torch.zeros(feature_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

