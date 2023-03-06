# This script is used to filter out static-camera frames.
import os.path

import numpy as np
import cv2
from path import Path
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Selecting video frames for training sc_depth')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--data_file_list', required=True)
    parser.add_argument('--save_file_list', required=True)

    args = parser.parse_args()
    return args

def compute_movement_ratio(frame1, frame2):
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h, w = frame1_gray.shape
    diff = np.abs(frame1_gray - frame2_gray)
    ratio = (diff > 10).sum() / (h*w)
    return ratio

def generate_index(scene):

    images = sorted(scene.files('*.jpg'))

    index = [0]
    for idx in range(1, len(images)):

        frame1 = cv2.imread(images[index[-1]])
        frame2 = cv2.imread(images[idx])

        move_ratio = compute_movement_ratio(frame1, frame2)
        if move_ratio < 0.5:
            continue
        index.append(idx)

    print(len(images), len(index))
    return index

def generate_list_cam(image_path, image_list, save_file_list):
    with open(image_list, 'r') as f:
        image_names = [f.strip() for f in f.readlines()]
    monodepth2_list = [image_names[0]]
    result_list=[]
    print("all image:", len(image_names))
    for idx in tqdm(range(1,len(image_names))):
        if idx % 1000 ==0:
            print("idx: ", idx)
            print("ratio > 0.5 image: ", len(monodepth2_list))

        image_name1 = os.path.join(image_path, image_names[idx - 1].split()[0])
        image_name2 = os.path.join(image_path, image_names[idx].split()[0])
        frame1 = cv2.imread(image_name1)
        frame2 = cv2.imread(image_name2)
        if frame1 is None or frame2 is None:
            continue
        move_ratio = compute_movement_ratio(frame1, frame2)

        if move_ratio < 0.5:
            continue
        else:
            monodepth2_list.append(image_names[idx])
    print("ratio > 0.5 all images:", len(monodepth2_list))

    if len(monodepth2_list) > 1:
        for idx in tqdm(range(1,len(monodepth2_list) - 1)):
            # before = os.path.join(*(monodepth2_list[idx - 1].split('/')[:-3]))
            # current = os.path.join(*(monodepth2_list[idx].split('/')[:-3]))
            # after = os.path.join(*(monodepth2_list[idx + 1].split('/')[:-3]))
            # before = os.path.join(*(monodepth2_list[idx - 1].split('_')[:-1]))
            # current = os.path.join(*(monodepth2_list[idx].split('_')[:-1]))
            # after = os.path.join(*(monodepth2_list[idx + 1].split('_')[:-1]))
            before = os.path.join(*(monodepth2_list[idx - 1].split('/')[:-1]))
            current = os.path.join(*(monodepth2_list[idx].split('/')[:-1]))
            after = os.path.join(*(monodepth2_list[idx + 1].split('/')[:-1]))
            print(before,current,after)

            if before == current and current == after:

                result_list.append([monodepth2_list[idx - 1], monodepth2_list[idx],monodepth2_list[idx + 1]])

        print("valid image: ", len(result_list))

    print("write file list in path: ", save_file_list)
    with open(save_file_list, 'w') as f:
        for img_name in result_list:
            if 'cam0' in img_name[0] and 'cam0' in img_name[1] and 'cam0' in img_name[2]:
                img_name.append('l')
            elif 'cam1' in img_name[0] and 'cam1' in img_name[1] and 'cam1' in img_name[2]:
                img_name.append('r')
            elif 'left' in img_name[0] and 'left' in img_name[1] and 'left' in img_name[2]:
                img_name.append('l')
            elif 'right' in img_name[0] and 'right' in img_name[1] and 'right' in img_name[2]:
                img_name.append('r')
                continue

            f.write(' '.join(img_name) + '\n')

def main():

    args = parse_args()

    generate_list_cam(args.dataset_dir, args.data_file_list, args.save_file_list)


if __name__ == '__main__':
    main()
