import os
import random

if __name__ == "__main__":
    for mode in ["train", "val"]:
        cs_f = open(os.path.join("/home/gaof/workspace/monodepth2/splits/cs_motion", "{}_files.txt".format(mode)), 'r')
        k_f = open(os.path.join("/home/gaof/workspace/monodepth2/splits/kitti_motion", "{}_files.txt".format(mode)), 'r')

        lines = cs_f.readlines() + k_f.readlines()
        random.shuffle(lines)
        with open("{}_files.txt".format(mode), 'w') as f:
            f.writelines(lines)
        cs_f.close()
        k_f.close()