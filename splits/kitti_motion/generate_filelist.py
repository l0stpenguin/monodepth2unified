from path import Path
import random
import os

if __name__ == "__main__":
    full_list_dir = Path("/home/share/lmDISK/datasets/motion/kitti_motion")
    for mode in ["train", "val"]:
        filelist = []
        data_dir = full_list_dir/mode
        with open("{}_files.txt".format(mode), 'w') as f:
            for scene_dir in data_dir.dirs():
                temp_list = []
                for side in ["left"]:
                    side_dir = scene_dir/side
                    count = 0
                    for img_path in sorted(side_dir.files("*.png")):
                        img_dir = scene_dir.split(data_dir/'')[-1]
                        img_name = img_path.split('/')[-1].split('.')[0]
                        img_idx = int(img_name)
                        img_side = side[0]
                        label_path = scene_dir/'label'/'{}.png'.format(img_idx)
                        if not os.path.exists(scene_dir/'label'/'{}.png'.format(img_idx)):
                            print(scene_dir/'label'/'{}.png'.format(img_idx))
                            continue
                        if not os.path.exists(scene_dir/side/'{:010d}.png'.format(img_idx+1)):
                            print(scene_dir/side/'{:010d}.png'.format(img_idx+1))
                            continue
                        img_string = "kitti_motion {} {} {}\n".format(img_dir, img_idx, img_side)
                        temp_list.append(img_string)
                filelist += temp_list
            random.shuffle(filelist)
            f.writelines(filelist)
