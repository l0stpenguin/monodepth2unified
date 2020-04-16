from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .net_utils import ResnetEncoder, PoseDecoder

class PoseModel(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(PoseModel, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=num_layers, 
            pretrained=pretrained, 
            num_input_images=2)
        self.decoder = PoseDecoder(
            num_ch_enc=self.encoder.num_ch_enc,
            num_input_features=1)

    def forward(self, x):
        features = [self.encoder(x)]
        axisangle, translation = self.decoder(features)
        return axisangle, translation