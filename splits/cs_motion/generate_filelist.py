from path import Path
import os
import random
from shutil import copyfile
import tqdm

if __name__ == "__main__":
    label_root = Path("/home/share/lmDISK/datasets/motion/Cityscapes-Motion")
    data_root = Path("/home/share/lmDISK/datasets/cityscapes/leftImg8bit_sequence")
    save_root = Path("/home/share/lmDISK/datasets/motion/cs_motion")
    for mode in ["train", "val"]:
        filelist = []
        label_dir = label_root/mode
        data_dir = data_root/mode
        save_dir = save_root/mode
        save_dir.mkdir_p()
        for label_scene_dir in label_dir.dirs():
            scene_name = label_scene_dir.split('/')[-1]
            print(scene_name)
            data_scene_dir = data_dir/scene_name
            save_scene_dir = save_dir/scene_name
            save_scene_dir.mkdir_p()
            save_label_dir = save_scene_dir/'label'
            save_image_dir = save_scene_dir/'left'
            save_label_dir.mkdir_p()
            save_image_dir.mkdir_p()
            for label_path in tqdm.tqdm(sorted(label_scene_dir.files("*.png"))):
                label_name = label_path.split('/')[-1].split('.')[0]
                seq_idx = int(label_name.split('_')[1])
                img_idx = int(label_name.split('_')[2])
                data_path = data_scene_dir/"{}_{:06d}_{:06d}_leftImg8bit.png".format(scene_name, seq_idx, img_idx)
                data_b_path = data_scene_dir/"{}_{:06d}_{:06d}_leftImg8bit.png".format(scene_name, seq_idx, img_idx-1)
                data_f_path = data_scene_dir/"{}_{:06d}_{:06d}_leftImg8bit.png".format(scene_name, seq_idx, img_idx+1)
                if os.path.exists(data_path) and os.path.exists(data_b_path) and os.path.exists(data_f_path):
                    filelist.append("cs_motion {} {}_{} {}\n".format(scene_name, seq_idx, img_idx, 'l'))
                    copyfile(data_path, save_image_dir/"{:06d}_{:06d}.png".format(seq_idx, img_idx))
                    copyfile(data_b_path, save_image_dir/"{:06d}_{:06d}.png".format(seq_idx, img_idx-1))
                    copyfile(data_f_path, save_image_dir/"{:06d}_{:06d}.png".format(seq_idx, img_idx+1))
                    copyfile(label_path, save_label_dir/"{}_{}.png".format(seq_idx, img_idx))
                else:
                    print(label_path)
            print('\tDone!')
        random.shuffle(filelist)
        with open("{}_files.txt".format(mode), 'w') as f:
            f.writelines(filelist)