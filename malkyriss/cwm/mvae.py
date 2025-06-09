
import torch
import torch.nn as nn


class CounterFactualModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def reconstruct(self, x0, xt):
        return