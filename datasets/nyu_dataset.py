from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset

class NYUDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        fx = 5.1885790117450188e+02 / 561
        fy = 5.1946961112127485e+02 / 427
        cx = 3.2558244941119034e+02 / 561
        cy = 2.5373616633400465e+02 / 427
        self.K = np.array([[fx, 0, cx, 0],
                           [0, fy, cy, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 0]], dtype=np.float32)
        self.full_res_shape = (427, 561)
    
    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, f_str)
        color = self.loader(image_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color