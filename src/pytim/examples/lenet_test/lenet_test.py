# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim.lib import *

def construct_tensors(engine):
    lenet_weight_data = np.load("./examples/lenet_test/lenet.npy")
    tensor_info = {}
    tensor_name = "input"
    tensor_info["shape"] = [28, 28, 1, 1]
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "INPUT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.00390625
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv1_weight"
    tensor_info["shape"] = [5, 5, 1, 20]
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.00336234
    tensor_info["quant_info"]["zero_point"] = 119
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[0:500]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv1_bias"
    tensor_info["shape"] = [20,]
    tensor_info["data_type"] = "INT32"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 1.313e-05
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[500:580]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv1_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.01928069
    tensor_info["quant_info"]["zero_point"] = 140
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "pool1_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.01928069
    tensor_info["quant_info"]["zero_point"] = 140
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv2_weight"
    tensor_info["shape"] = [5, 5, 20, 50]
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.0011482
    tensor_info["quant_info"]["zero_point"] = 128
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[580:25580]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv2_bias"
    tensor_info["shape"] = [50,]
    tensor_info["data_type"] = "INT32"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 2.214e-05
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[25580:25780]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "conv2_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.04075872
    tensor_info["quant_info"]["zero_point"] = 141
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "pool2_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.04075872
    tensor_info["quant_info"]["zero_point"] = 141
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "fc3_weight"
    tensor_info["shape"] = [800, 500]
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.00073548
    tensor_info["quant_info"]["zero_point"] = 130
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[25780:425780]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "fc3_bias"
    tensor_info["shape"] = [500,]
    tensor_info["data_type"] = "INT32"
    tensor_info["attribute"] = "CONSTANT"    
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 2.998e-05
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[425780:427780]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "fc3_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.01992089
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "relu_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.01992089
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)


    tensor_info = {}
    tensor_name = "fc4_weight"
    tensor_info["shape"] = [500, 10]
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.00158043
    tensor_info["quant_info"]["zero_point"] = 135
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[427780:432780]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "fc4_bias"
    tensor_info["shape"] = [10,]
    tensor_info["data_type"] = "INT32"
    tensor_info["attribute"] = "CONSTANT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 3.148e-05
    tensor_info["quant_info"]["zero_point"] = 0
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    tensor_info["data"] = lenet_weight_data[432780:432820]
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "fc4_output"
    tensor_info["shape"] = []
    tensor_info["data_type"] = "UINT8"
    tensor_info["attribute"] = "TRANSIENT"
    tensor_info["quant_info"] = {}
    tensor_info["quant_info"]["scale"] = 0.06251489
    tensor_info["quant_info"]["zero_point"] = 80
    tensor_info["quant_info"]["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)

    tensor_info = {}
    tensor_name = "output"
    tensor_info["shape"] = [10, 1]
    tensor_info["data_type"] = "FLOAT32"
    tensor_info["attribute"] = "OUTPUT"
    assert engine.create_tensor(tensor_name, tensor_info), "construct tensor {} fail!".format(tensor_name)


def construct_operations(engine):
    op_info = {}
    op_name = "conv1"
    op_info["op_type"] = "Conv2d"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}
    op_info["op_attr"]["weights"] = 20
    op_info["op_attr"]["padding"] = "VALID"
    op_info["op_attr"]["ksize"] = [5, 5]
    op_info["op_attr"]["stride"] = [1, 1]
    op_info["op_attr"]["dilation"] = [1, 1]
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["input", "conv1_weight", "conv1_bias"]), \
        "operation {} bind inputs fail!".format(op_name)
    print("phase 3")
    assert engine.bind_outputs(op_name, ["conv1_output", ]), \
        "operation {} bind outputs fail!".format(op_name)


    op_info = {}
    op_name = "pool1"
    op_info["op_type"] = "Pool2d"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}    
    op_info["op_attr"]["type"] = "MAX"
    op_info["op_attr"]["padding"] = "NONE"
    op_info["op_attr"]["ksize"] = [2, 2]
    op_info["op_attr"]["stride"] = [2, 2]
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["conv1_output", ]), \
        "operation {} bind inputs fail!".format(op_name)
    assert engine.bind_outputs(op_name, ["pool1_output", ]), \
        "operation {} bind outputs fail!".format(op_name)

    op_info = {}
    op_name = "conv2"
    op_info["op_type"] = "Conv2d"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}    
    op_info["op_attr"]["weights"] = 50
    op_info["op_attr"]["padding"] = "VALID"
    op_info["op_attr"]["ksize"] = [5, 5]
    op_info["op_attr"]["stride"] = [1, 1]
    op_info["op_attr"]["dilation"] = [1, 1]
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["pool1_output", "conv2_weight", "conv2_bias"]), \
        "operation {} bind inputs fail!".format(op_name)
    assert engine.bind_outputs(op_name, ["conv2_output", ]), \
        "operation {} bind outputs fail!".format(op_name)


    op_info = {}
    op_name = "pool2"
    op_info["op_type"] = "Pool2d"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}    
    op_info["op_attr"]["type"] = "MAX"
    op_info["op_attr"]["padding"] = "NONE"
    op_info["op_attr"]["ksize"] = [2, 2]
    op_info["op_attr"]["stride"] = [2, 2]
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["conv2_output", ]), \
        "operation {} bind inputs fail!".format(op_name)
    assert engine.bind_outputs(op_name, ["pool2_output", ]), \
        "operation {} bind outputs fail!".format(op_name)


    op_info = {}
    op_name = "fc3"
    op_info["op_type"] = "FullyConnected"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}
    op_info["op_attr"]["axis"] = 2
    op_info["op_attr"]["weights"] = 500
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["pool2_output", "fc3_weight", "fc3_bias"]), \
        "operation {} bind inputs fail!".format(op_name)
    assert engine.bind_outputs(op_name, ["fc3_output", ]), \
        "operation {} bind outputs fail!".format(op_name)


    op_info = {}
    op_name = "relu"
    op_info["op_type"] = "Activation"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}
    op_info["op_attr"]["activation_type"] = "Relu"
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_input(op_name, "fc3_output"), \
        "operation {} bind input fail!".format(op_name)
    assert engine.bind_output(op_name, "relu_output"), \
        "operation {} bind output fail!".format(op_name)


    op_info = {}
    op_name = "fc4"
    op_info["op_type"] = "FullyConnected"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}
    op_info["op_attr"]["axis"] = 0
    op_info["op_attr"]["weights"] = 10
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_inputs(op_name, ["relu_output", "fc4_weight", "fc4_bias"]), \
        "operation {} bind inputs fail!".format(op_name)
    assert engine.bind_outputs(op_name, ["fc4_output", ]), \
        "operation {} bind outputs fail!".format(op_name)


    op_info = {}
    op_name = "softmax"
    op_info["op_type"] = "Softmax"
    op_info["op_name"] = op_name
    op_info["op_attr"] = {}
    op_info["op_attr"]["beta"] = 1.0
    op_info["op_attr"]["axis"] = 0
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    assert engine.bind_input(op_name, "fc4_output"), \
        "operation {} bind input fail!".format(op_name)
    assert engine.bind_output(op_name, "output"), \
        "operation {} bind output fail!".format(op_name)



if __name__ == "__main__":
    engine = timvx_engine("lenet")
    assert engine.create_graph(), "engine create grah fail!"
    print("1 construct tensors begin....")
    construct_tensors(engine)
    print("1 construct tensors end....")

    print("2 construct operations begin....")
    construct_operations(engine)
    print("2 construct operations end....")

    print("3 compile graph begin....")
    assert engine.compile_graph(), "compile graph fail...."
    print("3 compile graph end....")

    print("4 set input begin....")
    lenet_input_data = np.load("./examples/lenet_test/input.npy").reshape((28, 28, 1, 1))
    assert engine.copy_data_to_tensor("input", lenet_input_data), "set input fail...."
    print("4 set input end....")

    print("5 run graph begin....")
    assert engine.run_graph(), "run graph fail...."
    print("5 run graph end....")

    print("6 get output begin....")
    output_data = np.zeros((10,1)).astype(np.float32)
    assert engine.copy_data_from_tensor("output", output_data), "get output fail...."
    print("6 get output end....")
    print(output_data)