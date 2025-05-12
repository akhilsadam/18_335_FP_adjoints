from .sine import Sine, AddSine
import torch.nn as nn

act={
    'sine': Sine,
    'add_sine': AddSine,
    'relu': nn.ReLU,
    'none': nn.Identity,
}