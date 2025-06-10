
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mvae import CounterFactualModel

__all__ = ["extract_segments"]

def extract_segments(model : CounterFactualModel):
    return