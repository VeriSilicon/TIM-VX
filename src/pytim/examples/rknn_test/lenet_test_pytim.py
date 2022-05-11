# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *


if __name__ == "__main__":
    # convert rknn to timvx engine
    rknn_file_name = "./examples/rknn_test/lenet.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=True)
    
    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."
    
    # set engine's input
    lenet_input_data = np.load("./examples/rknn_test/lenet_input.npy").reshape((1, 28, 28, 1))
    # lenet_input_data = np.load("./examples/rknn_test/lenet_random_input.npy").reshape((1, 28, 28, 1))
    lenet_input_data = lenet_input_data.transpose((2,1,3,0)) # convert nhwc to whcn
    assert engine.copy_data_to_tensor("norm_tensor:1", lenet_input_data), "set input fail...."
    
    # run engine's graph
    assert engine.run_graph(), "run graph fail...."

    # get output
    output_data = np.zeros((10,1)).astype(np.uint8)
    assert engine.copy_data_from_tensor("norm_tensor:0", output_data), "get output fail...."
    print(output_data)
