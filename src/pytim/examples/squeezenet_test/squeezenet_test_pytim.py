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
    rknn_file_name = "./examples/squeezenet_test/squeezenet.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=True)

    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."

    # set engine's input
    input_data = cv2.imread("./examples/squeezenet_test/squeezenet_test.jpg")
    input_data = input_data.transpose((2,0,1)).reshape(224,224,3,1)
    assert engine.copy_data_to_tensor("norm_tensor:1", input_data), "set input fail...."

    # run engine's graph
    assert engine.run_graph(), "run graph fail...."

    # get output
    output_data = np.zeros((1, 1, 1000, 1)).astype(np.float16)
    assert engine.copy_data_from_tensor("norm_tensor:0", output_data), "get output fail...."
    output_data = output_data.reshape((1000,)).astype(np.float32)
    print("**********************")
    top_k=5
    top_k_idx=output_data.argsort()[::-1][0:top_k]
    print(output_data[top_k_idx])
