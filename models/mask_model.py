from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from .net_utils import ResnetEncoder, MaskDecoder

class MaskModel(nn.Module):
    def __init__(self, num_class=3, num_layers=18, num_scales=4, pretrained=True):
        super(MaskModel, self).__init__()
        self.num_scales = num_scales
        self.num_class = num_class
        self.encoder = ResnetEncoder(
            num_layers=num_layers, 
            pretrained=pretrained,
            num_input_images=2)
        self.decoder = MaskDecoder(
            num_ch_enc=self.encoder.num_ch_enc, 
            scales=range(self.num_scales),
            num_output_channels=self.num_class)

    def forward(self, img, frame_id=0):
        features = self.encoder(img)
        outputs = self.decoder(features, frame_id)
        return outputs