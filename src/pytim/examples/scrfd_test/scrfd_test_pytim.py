# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *
from example.scrfd_test.utils import letterbox, decode

if __name__ == "__main__":
    # convert rknn to timvx engine
    rknn_file_name = "./examples/scrfd_test/scrfd.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=False)

    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."

    # set engine's input
    src_img = cv2.imread("./examples/scrfd_test/scrfd_test.jpg")
    img,ratio,pad = letterbox(src_img)
    input_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1)
    input_data = (input_data - 127.5) / 128.0
    input_data = np.round(input_data / 0.0078125 + 128).astype(np.uint8).reshape((320, 480, 3, 1))
    assert engine.copy_data_to_tensor("norm_tensor:9", input_data), "set input fail...."

    # run engine's graph
    assert engine.run_graph(), "run graph fail...."

    # get output
    outputs = []
    output_data_0 = np.zeros((1, 4800, 1)).astype(np.uint8)
    output_data_1 = np.zeros((1, 1200, 1)).astype(np.uint8)
    output_data_2 = np.zeros((1, 300, 1)).astype(np.uint8)
    output_data_3 = np.zeros((4, 4800, 1)).astype(np.uint8)
    output_data_4 = np.zeros((4, 1200, 1)).astype(np.uint8)
    output_data_5 = np.zeros((4, 300, 1)).astype(np.uint8)
    output_data_6 = np.zeros((10, 4800, 1)).astype(np.uint8)
    output_data_7 = np.zeros((10, 1200, 1)).astype(np.uint8)
    output_data_8 = np.zeros((10, 300, 1)).astype(np.uint8)
    assert engine.copy_data_from_tensor("norm_tensor:0", output_data_0), "get output 0 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:1", output_data_1), "get output 1 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:2", output_data_2), "get output 2 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:3", output_data_3), "get output 3 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:4", output_data_4), "get output 4 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:5", output_data_5), "get output 5 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:6", output_data_6), "get output 6 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:7", output_data_7), "get output 7 fail...."
    assert engine.copy_data_from_tensor("norm_tensor:8", output_data_8), "get output 8 fail...."
    output_data_0 = ((output_data_0 - 0.0) * 0.00328024267219007)
    output_data_1 = ((output_data_1 - 0.0) * 0.003378970082849264)
    output_data_2 = ((output_data_2 - 0.0) * 0.0034150767605751753)
    output_data_3 = ((output_data_3 - 1.0) * 0.017149630934000015).reshape((1,4800,4))
    output_data_4 = ((output_data_4 - 0.0) * 0.03239903599023819).reshape((1,1200,4))
    output_data_5 = ((output_data_5 - 0.0) * 0.021522123366594315).reshape((1,300,4))
    output_data_6 = ((output_data_6 - 117.0) * 0.02149081602692604).reshape((1,4800,10))
    output_data_7 = ((output_data_7 - 118.0) * 0.03708931431174278).reshape((1,1200,10))
    output_data_8 = ((output_data_8 - 118.0) * 0.025925135239958763).reshape((1,300,10))
    outputs.append(output_data_0)
    outputs.append(output_data_1)
    outputs.append(output_data_2)
    outputs.append(output_data_3)
    outputs.append(output_data_4)
    outputs.append(output_data_5)
    outputs.append(output_data_6)
    outputs.append(output_data_7)
    outputs.append(output_data_8)

    # post process
    print("**********************")
    rknn_detect_faces = decode(ratio, pad, src_img, img, outputs, 0.4, 0.45)
    print("detect {} faces".format(len(rknn_detect_faces)))
    print("face location:{}".format(rknn_detect_faces))
