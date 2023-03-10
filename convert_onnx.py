#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2023/3/9 下午6:32
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from SC_Depth import SC_Depth
import torch
import cv2
import numpy as np
from onnxmodel import ONNXModel


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--output", type=str, help="output model path")
    parser.add_argument("--image", type=str, help="test image file")
    # model
    parser.add_argument('--model_version', type=str,
                        default='v1', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--resnet_layers', type=int, default=18)

    args = parser.parse_args()
    return args

def test_onnx(img_path, model_file):
    model = ONNXModel(model_file)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 384), cv2.INTER_LANCZOS4)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    img = img / 255
    img = np.subtract(img, mean)
    img = np.divide(img, std)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    dis_array = output[0][0][0]
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    showImg = cv2.resize(dis_array, (dis_array.shape[-1], dis_array.shape[0]))
    showImg = cv2.applyColorMap(cv2.convertScaleAbs(showImg, 1), cv2.COLORMAP_PARULA)
    cv2.imwrite("onnx_result.jpg", showImg)

def main():
    args = GetArgs()

    system = SC_Depth(args)

    # load ckpts
    system = system.load_from_checkpoint(args.model, strict=False)

    device = torch.device("cuda")
    model = system.depth_net
    model.to(device)
    model.eval()

    onnx_input = torch.rand(1, 3, 384, 640)
    onnx_input = onnx_input.to("cuda:0")
    torch.onnx.export(model,
                      onnx_input,
                      args.output,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])

    test_onnx(args.image, args.output)


if __name__ == '__main__':
    main()
