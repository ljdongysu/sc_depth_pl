import cv2
import os
import numpy as np
import argparse
from tqdm.contrib import tzip

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def get_file_name(file_name, depth_dir):
    with open(file_name, 'r') as f:
        file_lists = f.readlines()
    depth_list = []
    for image_name in file_lists:
        image_current = image_name.split()[1]
        image_depth_path = os.path.join(depth_dir, image_current)
        depth_list.append(image_depth_path)

    return depth_list, file_lists
def svae_depth_by_var(depth_image_list, dest_dir):
    image_index = 0
    for depth_image_path in depth_image_list:
        print(depth_image_path)
        depth_image = cv2.imread(depth_image_path)
        depth_var = np.var(depth_image)
        dest_depth_path = dest_dir + "/" + str(image_index) + "_" +str(int(depth_var))+"_"+ depth_image_path.split("/")[-1]
        MkdirSimple(dest_depth_path)
        print(dest_depth_path)
        cv2.imwrite(dest_depth_path, depth_image)
        image_index += 1

def GetArgs():
    parser = argparse.ArgumentParser(description='calculate depth image var for filter file lists')
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--save_file_list', type=str, required=True)
    parser.add_argument('--depth_path', type=str, required=True)
    parser.add_argument('--var_value', type=float, default=300)
    parser.add_argument('--save_depth_var_name', type=str, default="")
    args = parser.parse_args()

    return args

def save_file_lists(depth_image_list, file_lists, var_value, save_file_list):
    image_save = 0
    with open(save_file_list, "w") as f:
        for depth_image_path, file_list in tzip(depth_image_list, file_lists):
            depth_image = cv2.imread(depth_image_path)
            depth_var = np.var(depth_image)
            if depth_var > var_value:
                image_save += 1
                f.write(file_list)
                if image_save % 500 == 0:
                    print("current images: {}".format(image_save))



if __name__ == '__main__':
    args = GetArgs()
    print("start !")
    depth_image_list, file_lists = get_file_name(args.file_list, args.depth_path)
    if args.save_depth_var_name != "":
        svae_depth_by_var(depth_image_list, args.save_depth_var_name)

    save_file_lists(depth_image_list, file_lists, args.var_value, args.save_file_list)

