import argparse
import os
import torch
from tqdm import tqdm
import cv2

from test_image_i18R import write_info
from test_image_i18R import best_scale
from test_image_i18R import GetImages
from test_image_i18R import MkdirSimple
from test_image_i18R import best_scale_filter
from test_image_i18R import DATA_TYPE, DEPTH_POINT, SCALE_POINT, MIN_POINT_ONE_IMAGE, IMAGE_GRAY_SCALE_PATH, MIN_SCALE

def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    return args

def caculate_scale(image_name, predict_np_gray_scale):
    image_split = image_name.split('/')[-1]
    IMAGE_GRAY_SCALE_PATH[image_split] = image_name
    if image_split in DEPTH_POINT:
        for point in DEPTH_POINT[image_split]:
            if (len(predict_np_gray_scale.shape)) == 3:
                SCALE_POINT[image_split][point] = predict_np_gray_scale[point[1], point[0]][0] / DEPTH_POINT[image_split][point]
            elif (len(predict_np_gray_scale.shape)) == 2:
                SCALE_POINT[image_split][point] = predict_np_gray_scale[point[1], point[0]] / DEPTH_POINT[image_split][point]

def get_point_value(output_dir):
    image_point = {}
    for image_name in DEPTH_POINT:
        output_gray_scale = IMAGE_GRAY_SCALE_PATH[image_name]
        output_gray_scale_save = output_dir + "/" + image_name
        # output_gray_scale_save_all = output_gray_scale.replace("gray_scale", "gray_scale_point_all.png")
        MkdirSimple(output_gray_scale_save)
        # MkdirSimple(output_gray_scale_save_all)

        image = cv2.imread(output_gray_scale)
        # image_all = image.copy()
        min_point = MIN_POINT_ONE_IMAGE[image_name]
        image_point[image_name] = []
        for point in DEPTH_POINT[image_name]:
            gt_value = DEPTH_POINT[image_name][point]
            scale = SCALE_POINT[image_name][point]
            predict_value = DEPTH_POINT[image_name][point] * SCALE_POINT[image_name][point]
            scaled_one_image_point_value = predict_value / SCALE_POINT[image_name][min_point]
            scaled_all_image_point_value = predict_value / MIN_SCALE[-1]
            distance_one = abs(scaled_one_image_point_value - gt_value)
            distance_all = abs(scaled_all_image_point_value - gt_value)
            image = cv2.circle(image, point, 1, (0, 0, 255, 255), -1)
            image = cv2.putText(image, str(int(scaled_one_image_point_value)), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1,
                        cv2.LINE_AA)

            # image_all = cv2.circle(image_all, point, 1, (0, 0, 255, 255), -1)
            # image_all = cv2.putText(image_all, str(int(scaled_all_image_point_value)), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #                     (0, 0, 255), 1,
            #                     cv2.LINE_AA)

            # print("point: {}, gt_value: {}, predict_value: {}, scale_one: {}, scale_value: {}, scale_all: {}, scaled_all_image_point_value: {}".format(point, gt_value, predict_value, scale, scaled_one_image_point_value, MIN_SCALE[0], scaled_all_image_point_value))
            # print("point: {}, distance_one: {}, distance_all: {}".format(point, distance_one, distance_all))

            image_point[image_name].append([image_name,point, gt_value, predict_value, scale, MIN_SCALE[-1], scaled_one_image_point_value, scaled_all_image_point_value, distance_one, distance_all])
        cv2.imwrite(output_gray_scale_save, image)
        # cv2.imwrite(output_gray_scale_save_all, image_all)
    return image_point

@torch.no_grad()
def main():
    args = GetArgs()

    for k in DATA_TYPE:
        left_files, root_len = GetImages(args.data_path, k)

        if len(left_files) != 0:
            break

    for left_image_file in tqdm(left_files):
        if not os.path.exists(left_image_file):
            continue

        output_gray_scale = left_image_file
        predict_np_gray_scale = cv2.imread(output_gray_scale)
        caculate_scale(output_gray_scale, predict_np_gray_scale)

    best_scale()
    print("args.output: ", args.output)

    depth_info = get_point_value(args.output)
    write_info(depth_info, args.output + "/result.csv")

    depth_info_filter = best_scale_filter(1)
    write_info(depth_info_filter, args.output + "/result" + "_" + str(1) + "m.csv")

    depth_info_filter = best_scale_filter(1.5)
    write_info(depth_info_filter, args.output + "/result" + "_" + str(1.5) + "m.csv")

if __name__ == '__main__':
    main()