# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
cwd_path = os.getcwd()
sys.path.append(cwd_path)
from pytim import *

def construct_tensors(engine):
    lenet_weight_data = np.load("./examples/lenet_test/lenet.npy")
    tensor_name = "input"
    quant_info = {}
    quant_info["scale"] = 0.00390625
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "INPUT", [28, 28, 1, 1], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv1_weight"
    quant_info = {}
    quant_info["scale"] = 0.00336234
    quant_info["zero_point"] = 119
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[0:500]
    assert engine.create_tensor(tensor_name, "UINT8", "CONSTANT", [5, 5, 1, 20], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv1_bias"
    quant_info = {}
    quant_info["scale"] = 1.313e-05
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[500:580]
    assert engine.create_tensor(tensor_name, "INT32", "CONSTANT", [20,], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv1_output"
    quant_info = {}
    quant_info["scale"] = 0.01928069
    quant_info["zero_point"] = 140
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "pool1_output"
    quant_info = {}
    quant_info["scale"] = 0.01928069
    quant_info["zero_point"] = 140
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv2_weight"
    quant_info = {}
    quant_info["scale"] = 0.0011482
    quant_info["zero_point"] = 128
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[580:25580]
    assert engine.create_tensor(tensor_name, "UINT8", "CONSTANT", [5, 5, 20, 50], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv2_bias"
    quant_info = {}
    quant_info["scale"] = 2.214e-05
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[25580:25780]
    assert engine.create_tensor(tensor_name, "INT32", "CONSTANT", [50,], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "conv2_output"
    quant_info = {}
    quant_info["scale"] = 0.04075872
    quant_info["zero_point"] = 141
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "pool2_output"
    quant_info = {}
    quant_info["scale"] = 0.04075872
    quant_info["zero_point"] = 141
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc3_weight"
    quant_info = {}
    quant_info["scale"] = 0.00073548
    quant_info["zero_point"] = 130
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[25780:425780]
    assert engine.create_tensor(tensor_name, "UINT8", "CONSTANT", [800, 500], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc3_bias"
    quant_info = {}
    quant_info["scale"] = 2.998e-05
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[425780:427780]
    assert engine.create_tensor(tensor_name, "INT32", "CONSTANT", [500,], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc3_output"
    quant_info = {}
    quant_info["scale"] = 0.01992089
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "relu_output"
    quant_info = {}
    quant_info["scale"] = 0.01992089
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc4_weight"
    quant_info = {}
    quant_info["scale"] = 0.00158043
    quant_info["zero_point"] = 135
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[427780:432780]
    assert engine.create_tensor(tensor_name, "UINT8", "CONSTANT", [500, 10], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc4_bias"
    quant_info = {}
    quant_info["scale"] = 3.148e-05
    quant_info["zero_point"] = 0
    quant_info["quant_type"] = "ASYMMETRIC"
    data = lenet_weight_data[432780:432820]
    assert engine.create_tensor(tensor_name, "INT32", "CONSTANT", [10,], quant_info, data), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "fc4_output"
    quant_info = {}
    quant_info["scale"] = 0.06251489
    quant_info["zero_point"] = 80
    quant_info["quant_type"] = "ASYMMETRIC"
    assert engine.create_tensor(tensor_name, "UINT8", "TRANSIENT", [], quant_info), \
        "construct tensor {} fail!".format(tensor_name)

    tensor_name = "output"
    assert engine.create_tensor(tensor_name, "FLOAT32", "OUTPUT", [10, 1]), \
        "construct tensor {} fail!".format(tensor_name)


def construct_operations(engine):
    op_name = "conv1"
    op_inputs = ["input", "conv1_weight", "conv1_bias"]
    op_outputs = ["conv1_output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, weights=20, padding="VALID", ksize=[5,5], 
        stride=[1,1], dilation=[1,1], op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "pool1"
    op_inputs = ["conv1_output", ]
    op_outputs = ["pool1_output", ]
    op_info = ConstructPool2dOpConfig(op_name=op_name, type="MAX", padding="NONE", ksize=[2,2], 
        stride=[2,2], op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "conv2"
    op_inputs = ["pool1_output", "conv2_weight", "conv2_bias"]
    op_outputs = ["conv2_output", ]
    op_info = ConstructConv2dOpConfig(op_name=op_name, weights=50, padding="VALID", ksize=[5,5], 
        stride=[1,1], dilation=[1,1], op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "pool2"
    op_inputs = ["conv2_output", ]
    op_outputs = ["pool2_output", ]
    op_info = ConstructPool2dOpConfig(op_name=op_name, type="MAX", padding="NONE", ksize=[2,2], 
        stride=[2,2], op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "fc3"
    op_inputs = ["pool2_output", "fc3_weight", "fc3_bias"]
    op_outputs = ["fc3_output", ]
    op_info = ConstructFullyConnectedOpConfig(op_name=op_name, axis=2, weights=500, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "relu"
    op_inputs = ["fc3_output", ]
    op_outputs = ["relu_output", ]
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type="Relu", 
        op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "fc4"
    op_inputs = ["relu_output", "fc4_weight", "fc4_bias"]
    op_outputs = ["fc4_output", ]
    op_info = ConstructFullyConnectedOpConfig(op_name=op_name, axis=0, weights=10, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)

    op_name = "softmax"
    op_inputs = ["fc4_output", ]
    op_outputs = ["output", ]
    op_info = ConstructSoftmaxOpConfig(op_name=op_name, beta=1.0, axis=0, 
        op_inputs=op_inputs, op_outputs=op_outputs)
    print(op_info)
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)


if __name__ == "__main__":
    engine = Engine("lenet")
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