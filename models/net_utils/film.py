
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

class MLP(nn.Module):
    def __init__(self, input_length, output_length):
        super(MLP, self).__init__()
        self.layers = OrderedDict()
        self.layers[0] = nn.Linear(input_length, output_length)
        self.layers[1] = nn.Linear(output_length, output_length)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.layers[1](self.relu(self.layers[0](x)))

class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    gammas = gammas.repeat(1, 1, x.size(2), x.size(3))
    betas = betas.repeat(1, 1, x.size(2), x.size(3))
    return (gammas * x) + betas