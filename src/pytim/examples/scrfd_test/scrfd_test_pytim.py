# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *
from examples.scrfd_test.utils import letterbox, decode

if __name__ == "__main__":
    # convert rknn to timvx engine
    rknn_file_name = "./examples/scrfd_test/scrfd.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=False)

    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."

    # prepare engine's input
    src_img = cv2.imread("./examples/scrfd_test/scrfd_test.jpg")
    img,ratio,pad = letterbox(src_img)
    input_dict = {}
    input_dict["norm_tensor:9"] = img

    # run engine's graph and returen infer result
    outputs = engine.run_graph(input_dict)

    # post process
    print("**********************")
    rknn_detect_faces = decode(ratio, pad, src_img, img, outputs, 0.4, 0.45)
    print("detect {} faces".format(len(rknn_detect_faces)))
    print("face location:{}".format(rknn_detect_faces))

    # export engine's graph
    assert engine.export_graph("./examples/scrfd_test/scrfd.json", 
        "./examples/scrfd_test/scrfd.weight"), "export graph fail...."