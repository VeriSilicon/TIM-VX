# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *


if __name__ == "__main__":
    # convert rknn to timvx engine
    rknn_file_name = "./examples/lenet_test/lenet.rknn"
    convert = Rknn2TimVxEngine()
    engine = convert.convert_to_timvx(rknn_file_name, log_flag=False)
    
    # compile engine's graph
    assert engine.compile_graph(), "compile graph fail...."

    # prepare engine's input
    lenet_input_data = np.load("./examples/lenet_test/lenet_input.npy").reshape((28, 28, 1))
    input_dict = {}
    input_dict["norm_tensor:1"] = lenet_input_data

    # run engine's graph and returen infer result
    outputs = engine.run_graph(input_dict)
    print(outputs[0])
    
    # export engine's graph
    assert engine.export_graph("./examples/lenet_test/lenet.json", 
        "./examples/lenet_test/lenet.weight"), "export graph fail...."
