# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *


if __name__ == "__main__":
    # convert rknn to timvx engine
    rknn_file_name = "./examples/rknn_test/scrfd.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=False)

    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."

    # set engine's input
    input_data = cv2.imread("./examples/rknn_test/scrfd_test.jpg")
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB).transpose(2,0,1)
    input_data = input_data.reshape((1, 480, 320, 3)).transpose((2,1,3,0)) # convert nhwc to whcn
    input_data = (input_data - 127.5) / 128.0
    input_data = np.round(input_data / 0.0078125 + 128).astype(np.uint8)
    assert engine.copy_data_to_tensor("norm_tensor:9", input_data), "set input fail...."

    # run engine's graph
    assert engine.run_graph(), "run graph fail...."

    # get output
    output_data = np.zeros((1, 300,1)).astype(np.uint8)
    assert engine.copy_data_from_tensor("norm_tensor:2", output_data), "get output fail...."
    print(output_data[0,:,0])
