from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .net_utils import ResnetEncoder, DepthDecoder

class DepthModel(nn.Module):
    def __init__(self, num_layers=18, num_scales=4, pretrained=True):
        super(DepthModel, self).__init__()
        self.num_scales = num_scales
        self.encoder = ResnetEncoder(
            num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(self.num_scales))

    def forward(self, img, frame_id=0):
        features = self.encoder(img)
        outputs = self.decoder(features, frame_id)
        return outputs