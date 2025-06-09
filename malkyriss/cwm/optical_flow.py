import torch
import torch.nn as nn
from .mvae import CounterFactualModel

__all__ = ["extract_optical_flow"]

def extract_optical_flow(model : CounterFactualModel):
    return