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
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=False)

    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."


    # prepare engine's input
    input_data = cv2.imread("./examples/squeezenet_test/squeezenet_test.jpg")
    input_dict = {}
    input_dict["norm_tensor:1"] = input_data

    # run engine's graph and returen infer result
    outputs = engine.run_graph(input_dict)

    # post process get top 5 result
    output_data = outputs[0].reshape((1000,))
    print("**********************")
    top_k=5
    top_k_idx=output_data.argsort()[::-1][0:top_k]
    print(output_data[top_k_idx])

    # export engine's graph
    assert engine.export_graph("./examples/squeezenet_test/squeezenet.json", 
        "./examples/squeezenet_test/squeezenet.weight"), "export graph fail...."