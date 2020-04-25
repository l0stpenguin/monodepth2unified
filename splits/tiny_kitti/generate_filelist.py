from path import Path
import random

if __name__ == "__main__":
    full_list_dir = Path("/home/gaof/workspace/monodepth2/splits/eigen_zhou")
    for name in ["train", "val"]:
        full_list = open(full_list_dir/"{}_files.txt".format(name)).readlines()
        n = 10000 if name == "train" else 1000
        sub_list = random.sample(full_list, n)
        with open("{}_files.txt".format(name), 'w') as f:
            f.writelines(sub_list)
