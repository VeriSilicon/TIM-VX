# -*- coding: utf-8 -*-
import os
import numpy as np
from .rknn_parser import parse_rknn_model
from ..timvx import *

def convert_to_timvx_dtype(datatype:str):
    dtype_list = ["VSI_NN_TYPE_INT8", 
        "VSI_NN_TYPE_INT16",
        "VSI_NN_TYPE_INT32",
        "VSI_NN_TYPE_UINT8",
        "VSI_NN_TYPE_UINT16",
        "VSI_NN_TYPE_UINT32",
        "VSI_NN_TYPE_FLOAT16",
        "VSI_NN_TYPE_FLOAT32",
        "VSI_NN_TYPE_BOOL8"]
    if datatype == "VSI_NN_TYPE_INT8":
        return "INT8"
    elif datatype == "VSI_NN_TYPE_INT16":
        return "INT16"
    elif datatype == "VSI_NN_TYPE_INT32":
        return "INT32"
    elif datatype == "VSI_NN_TYPE_UINT8":
        return "UINT8"
    elif datatype == "VSI_NN_TYPE_UINT16":
        return "UINT16"
    elif datatype == "VSI_NN_TYPE_UINT32":
        return "UINT32"
    elif datatype == "VSI_NN_TYPE_FLOAT16":
        return "FLOAT16"
    elif datatype == "VSI_NN_TYPE_FLOAT32":
        return "FLOAT32"
    elif datatype == "VSI_NN_TYPE_BOOL8":
        return "BOOL8"
    else:
        assert False, "unsupported datatype {}, current only support {}".format(datatype, dtype_list)


def convert_timvx_dtype_to_np_dtype(datatype:str):
    dtype_list = ["INT8", 
        "INT16",
        "INT32",
        "UINT8",
        "UINT16",
        "UINT32",
        "FLOAT16",
        "FLOAT32",
        "BOOL8"]
    if datatype == "INT8":
        return np.int8
    elif datatype == "INT16":
        return np.int16
    elif datatype == "INT32":
        return np.int32
    elif datatype == "UINT8":
        return np.uint8
    elif datatype == "UINT16":
        return np.uint16
    elif datatype == "UINT32":
        return np.uint32
    elif datatype == "FLOAT16":
        return np.float16
    elif datatype == "FLOAT32":
        return np.float32
    elif datatype == "BOOL8":
        return np.bool8
    else:
        assert False, "unsupported datatype {}, current only support {}".format(datatype, dtype_list)


def convert_to_timvx_qnt_type(qnt_type:str):
    qnt_list = ["VSI_NN_QNT_TYPE_NONE", 
        "VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC",
        "VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC"]
    if qnt_type == "VSI_NN_QNT_TYPE_NONE":
        return "NONE"
    elif qnt_type == "VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC":
        return "SYMMETRIC"
    elif qnt_type == "VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC":
        return "ASYMMETRIC"
    else:
        assert False, "unsupported qnt type {}".format(qnt_type, qnt_list)


def convert_to_timvx_pool_type(pool_type:str):
    pool_list = ["VX_CONVOLUTIONAL_NETWORK_POOLING_MAX", 
        "VX_CONVOLUTIONAL_NETWORK_POOLING_AVG",
        "VX_CONVOLUTIONAL_NETWORK_POOLING_L2",
        "VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID"]
    if pool_type == "VX_CONVOLUTIONAL_NETWORK_POOLING_MAX":
        return "MAX"
    elif pool_type == "VX_CONVOLUTIONAL_NETWORK_POOLING_AVG":
        return "AVG"
    elif pool_type == "VX_CONVOLUTIONAL_NETWORK_POOLING_L2":
        return "L2"
    elif pool_type == "VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID":
        return "AVG_ANDROID"
    else:
        assert False, "unsupported pool type {}".format(pool_type, pool_list)


def convert_to_timvx_round_type(round_type:str):
    round_list = ["VSI_NN_ROUND_FLOOR", 
        "VSI_NN_ROUND_CEIL", 
        "VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR",
        "VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING"]
    if round_type == "VSI_NN_ROUND_FLOOR" or \
        round_type == "VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR":
        return "FLOOR"
    elif round_type == "VSI_NN_ROUND_CEIL" or \
        round_type == "VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING":
        return "CEILING"
    else:
        assert False, "unsupported round type {}".format(round_type, round_list)


def convert_to_timvx_rounding_policy(rounding_policy:str):
    rounding_policy_list = ["VX_ROUND_POLICY_TO_RTNE", 
        "VX_ROUND_POLICY_TO_ZERO",
        "VX_ROUND_POLICY_TO_NEAREST_EVEN"]
    if rounding_policy == "VX_ROUND_POLICY_TO_RTNE" or \
        rounding_policy == "VX_ROUND_POLICY_TO_NEAREST_EVEN":
        return "RTNE"
    elif rounding_policy == "VX_ROUND_POLICY_TO_ZERO":
        return "TO_ZERO"
    else:
        assert False, "unsupported rounding policy {}".format(rounding_policy, rounding_policy_list)


def convert_to_timvx_overflow_policy(overflow_policy:str):
    overflow_policy_list = ["VX_CONVERT_POLICY_SATURATE", 
        "VX_CONVERT_POLICY_WRAP"]
    if overflow_policy == "VX_CONVERT_POLICY_SATURATE":
        return "SATURATE"
    elif overflow_policy == "VX_CONVERT_POLICY_WRAP":
        return "WRAP"
    else:
        assert False, "unsupported overflow policy {}".format(overflow_policy, overflow_policy_list)


def convert_to_timvx_interpolation_type(interpolation_type:str):
    interpolation_type_list = ["VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR", 
        "VSI_NN_INTERPOLATION_BILINEAR",
        "VSI_NN_INTERPOLATION_AREA"]    
    if interpolation_type == "VSI_NN_INTERPOLATION_NEAREST_NEIGHBOR":
        return "NEAREST_NEIGHBOR"
    elif interpolation_type == "VSI_NN_INTERPOLATION_BILINEAR":
        return "BILINEAR"
    elif interpolation_type == "VSI_NN_INTERPOLATION_AREA":
        return "AREA"
    else:
        assert False, "unsupported interpolation type {}".format(interpolation_type, interpolation_type_list)


def format_rounding_policy(rounding_policy_dict):
    if "overflow_policy" in rounding_policy_dict.keys():
        policy = rounding_policy_dict["overflow_policy"]
        rounding_policy_dict["overflow_policy"] = convert_to_timvx_overflow_policy(policy)
    if "rounding_policy" in rounding_policy_dict.keys():
        policy = rounding_policy_dict["rounding_policy"]
        rounding_policy_dict["rounding_policy"] = convert_to_timvx_rounding_policy(policy)
    if "down_scale_size_rounding" in rounding_policy_dict.keys():
        policy = rounding_policy_dict["down_scale_size_rounding"]
        rounding_policy_dict["down_scale_size_rounding"] = convert_to_timvx_round_type(policy)

    return rounding_policy_dict


def format_resize_op_cfg(op_cfg:dict):
    if "type" in op_cfg.keys():
        op_cfg["type"] = convert_to_timvx_interpolation_type(op_cfg["type"])
    if "size" in op_cfg.keys():
        op_cfg["target_height"] = op_cfg["size"][1]
        op_cfg["target_width"] = op_cfg["size"][0]
        del op_cfg["size"]
    if "align_corners" in op_cfg.keys():
        op_cfg["align_corners"] = bool(op_cfg["align_corners"])
    if "half_pixel_centers" in op_cfg.keys():
        op_cfg["half_pixel_centers"] = bool(op_cfg["half_pixel_centers"])
    if "layout" in op_cfg.keys():
        op_cfg["layout"] = op_cfg["layout"]
    return op_cfg


def construct_activation_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    parameter = {}
    assert len(op_inputs) == 1, "rknn {} op should have 1 input".format(rknn_op_name)
    assert len(op_outputs) == 1, "rknn {} op should have 1 output".format(rknn_op_name)
    if rknn_op_name == "RELU":
        activation_type = "Relu"
    elif rknn_op_name == "SIGMOID":
        activation_type = "Sigmoid"
    else:
        assert False, "unspppoted activation type {}".format(rknn_op_name)
    op_info = ConstructActivationOpConfig(op_name=op_name, activation_type=activation_type, parameter=parameter,
        op_inputs=op_inputs, op_outputs=op_outputs)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_eltwise_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    parameter = {}
    assert len(op_inputs) == 2, "rknn {} op should have 2 input".format(rknn_op_name)
    assert len(op_outputs) == 1, "rknn {} op should have 1 output".format(rknn_op_name)
    if rknn_op_name == "ADD":
        eltwise_type = "Add"
    elif rknn_op_name == "MULTIPLY":
        eltwise_type = "Multiply"
    else:
        assert False, "unspppoted eltwise type {}".format(rknn_op_name)

    op_info = ConstructEltwiseOpConfig(op_name=op_name, eltwise_type=eltwise_type, parameter=parameter,
        op_inputs=op_inputs, op_outputs=op_outputs)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_conv2d_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 2 or len(op_inputs) == 3, "rknn conv2d op should have 2 or 3 input"
    assert len(op_outputs) == 1, "rknn conv2d op should have 1 output"
    assert "conv2d" in node_info["attribute"], "rknn conv2d op's attribute should have conv2d item"
    op_config = node_info["attribute"]["conv2d"]
    if "group" in op_config.keys():
        del op_config["group"]
    op_info = ConstructConv2dOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, **op_config)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_transpose_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 1, "rknn permute op should have 1 input"
    assert len(op_outputs) == 1, "rknn permute op should have 1 output"
    assert "permute" in node_info["attribute"], "rknn permute op's attribute should have permute item"
    perm = node_info["attribute"]["permute"]["perm"]

    op_info = ConstructTransposeOpConfig(op_name=op_name, perm=perm, op_inputs=op_inputs, op_outputs=op_outputs)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_fullyconnected_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 3, "rknn fcl op should have 3 input"
    assert len(op_outputs) == 1, "rknn fcl op should have 1 output"
    assert "fcl" in node_info["attribute"], "rknn fcl op's attribute should have fcl item"
    op_cfg = node_info["attribute"]["fcl"]
    if "axis" not in op_cfg.keys():
        op_cfg["axis"] = 0
    
    op_info = ConstructFullyConnectedOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, **op_cfg)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_concat_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) > 1, "rknn concat op input should greater than 1"
    assert len(op_outputs) == 1, "rknn concat op should have 1 output"
    assert "concat" in node_info["attribute"], "rknn permute op's attribute should have permute item"
    op_cfg = node_info["attribute"]["concat"]
    assert "axis" in node_info["attribute"]["concat"], "concat attr should have axis item"
    op_info = ConstructConcatOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, **op_cfg)

    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_softmax_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 1, "rknn softmax op should have 1 input"
    assert len(op_outputs) == 1, "rknn softmax op should have 1 output"
    assert "softmax" in node_info["attribute"], "rknn softmax op's attribute should have softmax item"
    assert "beta" in node_info["attribute"]["softmax"], "softmax attribute should have beta item"
    assert "axis" in node_info["attribute"]["softmax"], "softmax attribute should have axis item"
    beta = node_info["attribute"]["softmax"]["beta"]
    axis = node_info["attribute"]["softmax"]["axis"]

    op_info = ConstructSoftmaxOpConfig(op_name=op_name, beta=beta, axis=axis, op_inputs=op_inputs, op_outputs=op_outputs)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_reshape_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 1, "rknn reshape op should have 1 input"
    assert len(op_outputs) == 1, "rknn reshape op should have 1 output"
    assert "reshape" in node_info["attribute"], "rknn reshape op's attribute should have reshape item"
    size = node_info["attribute"]["reshape"]["size"]

    op_info = ConstructReshapeOpConfig(op_name=op_name, size=size, op_inputs=op_inputs, op_outputs=op_outputs)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_resize_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    assert len(op_inputs) == 1, "rknn resize op should have 1 input"
    assert len(op_outputs) == 1, "rknn resize op should have 1 output"
    assert "resize" in node_info["attribute"], "rknn resize op's attribute should have resize item"
    op_config = node_info["attribute"]["resize"]

    op_config = format_resize_op_cfg(op_config)
    op_info = ConstructResizeOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, **op_config)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_pool2d_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    node_info = rknn_model_info[node_index]
    op_name = node_info["name"]
    op_inputs = node_info["inputs"]
    op_outputs = node_info["outputs"]
    op_config = node_info["attribute"]["pool"]
    assert len(op_inputs) == 1, "rknn pool op should have 1 input"
    assert len(op_outputs) == 1, "rknn pool op should have 1 output"
    assert "pool" in node_info["attribute"], "rknn pool op's attribute should have pool item"

    if "type" in node_info["attribute"]["pool"].keys():
        pool_type = node_info["attribute"]["pool"]["type"]
        node_info["attribute"]["pool"]["type"] = convert_to_timvx_pool_type(pool_type)

    if "round_type" in node_info["attribute"]["pool"].keys():
        round_type = node_info["attribute"]["pool"]["round_type"]
        node_info["attribute"]["pool"]["round_type"] = convert_to_timvx_round_type(round_type)

    op_info = ConstructPool2dOpConfig(op_name=op_name, op_inputs=op_inputs, op_outputs=op_outputs, **op_config)
    if "vx" in node_info["attribute"].keys():
        op_info["rounding_policy"] = format_rounding_policy(node_info["attribute"]["vx"])
    if log_flag:
        print("construct {} op with info:\n{}".format(rknn_op_name, op_info))
    assert engine.create_operation(op_info), "construct operation {} fail!".format(op_name)
    return op_info


def construct_variable_op(rknn_op_name, rknn_model_info, node_index, engine, log_flag):
    return None


class ConstructTimVxOpFromRknn():
    op_construct_funcs = {}
    rknn_op_timvx_op_map = {}
    def register(self, rknn_op_name, timvx_op_name, op_func):
        if rknn_op_name not in self.rknn_op_timvx_op_map.keys():
            self.rknn_op_timvx_op_map[rknn_op_name] = timvx_op_name
            self.op_construct_funcs[timvx_op_name] = op_func
        else:
            print("already register {}".format(rknn_op_name))

    def construct_node(self, rknn_model_info, node_index, engine, log_flag = False):
        node_info = rknn_model_info[node_index]
        rknn_op_name = node_info['type']
        if rknn_op_name not in self.rknn_op_timvx_op_map.keys():
            assert False, "have not register {}".format(rknn_op_name)
        else:
            timvx_op_name = self.rknn_op_timvx_op_map[rknn_op_name]
            node_info = self.op_construct_funcs[timvx_op_name](rknn_op_name, rknn_model_info, node_index, engine, log_flag)
            return node_info


class Rknn2TimVxEngine():
    def __init__(self):
        self.constructor = ConstructTimVxOpFromRknn()
        self.constructor.register('SIGMOID', 'Activation', construct_activation_op)
        self.constructor.register('RELU', 'Activation', construct_activation_op)
        self.constructor.register('ADD', 'Eltwise', construct_eltwise_op)
        self.constructor.register('MULTIPLY', 'Eltwise', construct_eltwise_op)
        self.constructor.register('CONV2D', 'Conv2d', construct_conv2d_op)
        self.constructor.register('VARIABLE', 'Variable', construct_variable_op)
        self.constructor.register('RESHAPE', 'Reshape', construct_reshape_op)
        self.constructor.register('PERMUTE', 'Transpose', construct_transpose_op)
        self.constructor.register('RESIZE', 'Resize', construct_resize_op)
        self.constructor.register("POOL", "Pool2d", construct_pool2d_op)
        self.constructor.register("FCL", "FullyConnected", construct_fullyconnected_op)
        self.constructor.register("CONCAT", "Concat", construct_concat_op)
        self.constructor.register("SOFTMAX", "Softmax", construct_softmax_op)


    def format_rknn_model(self, rknn_model_info:dict):
        nodes_info = rknn_model_info["nodes"]
        substitute_map = {}
        for index in range(len(nodes_info)):
            node = nodes_info[index]
            if node["type"] == "VARIABLE":
                input = node["inputs"][0]
                out = node["outputs"][0]
                substitute_map[out] = input

        for node_index in range(len(nodes_info)):
            node = nodes_info[node_index]
            for index in range(len(node["inputs"])):
                input_name = node["inputs"][index]
                if input_name in substitute_map.keys():
                    node["inputs"][index] = substitute_map[input_name]

                    
        tensors_info = rknn_model_info["tensors"]
        for item in substitute_map.keys():
            del tensors_info[item]


    def creat_tensor(self, engine:Engine, tensor_name:str, tensor_attr:str, rk_tensor:dict, log_flag:bool=False):
        quant_info = {}
        tensor_shape = []
        if "shape" in rk_tensor.keys():
            tensor_shape = rk_tensor["shape"]
        tensor_dtype = convert_to_timvx_dtype(rk_tensor["dtype"]["vx_type"])
        qnt_type = convert_to_timvx_qnt_type(rk_tensor["dtype"]["qnt_type"])
        if qnt_type != "NONE":
            quant_info["scale"] = rk_tensor["dtype"]["scale"]
            quant_info["zero_point"] = rk_tensor["dtype"]["zero_point"]
            quant_info["quant_type"] = qnt_type
        np_data = np.array([])
        if "raw_data" in rk_tensor.keys():
            ori_shape = tensor_shape[::-1]
            np_dtype = convert_timvx_dtype_to_np_dtype(tensor_dtype)
            np_data = np.frombuffer(rk_tensor["raw_data"], dtype=np_dtype).reshape(ori_shape)
        if log_flag:
            print("********************************")
            print("construct tensor {} with:".format(tensor_name))
            print("tensor type    : {}".format(tensor_dtype))
            print("tensor attr    : {}".format(tensor_attr))
            print("tensor shape   : {}".format(tensor_shape))
            print("tensor qnt info: {}".format(quant_info))
            print("tensor data    : {}".format(np_data))
        
        assert engine.create_tensor(tensor_name, tensor_dtype, 
            tensor_attr, tensor_shape, quant_info, np_data), "creat tensor {} fail!".format(tensor_name)

        tensor_info = {}
        tensor_info["name"] = tensor_name
        tensor_info["dtype"] = convert_timvx_dtype_to_np_dtype(tensor_dtype)
        tensor_info["attr"] = tensor_attr
        tensor_info["shape"] = tensor_shape
        tensor_info["quant_info"] = quant_info
        if np_data.size != 0:
            tensor_info["data"] = np_data
        return tensor_info


    def construct_engine_tensors(self, rknn_model_info:dict, engine:Engine, log_flag:bool=False):
        tensors_info = rknn_model_info["inputs"]
        for index in range(len(tensors_info)):
            tensor_name = tensors_info[index]["name"]
            tensor_info = self.creat_tensor(engine, tensor_name, "INPUT", tensors_info[index]["tensor_info"], log_flag)
            if "url" in tensors_info[index]["tensor_info"].keys():
                tensor_info["alias"] = tensors_info[index]["tensor_info"]["url"]
            engine.add_inputs_info(tensor_name, tensor_info)

        tensors_info = rknn_model_info["tensors"]
        for tensor_name in tensors_info.keys():
            if "norm_tensor" in tensor_name:
                continue
            elif "const_tensor" in tensor_name:
                tensor_attr = "CONSTANT"
            else:
                tensor_attr = "TRANSIENT"
            tensor_info = self.creat_tensor(engine, tensor_name, tensor_attr, tensors_info[tensor_name], log_flag)
            engine.add_tensors_info(tensor_info)

        tensors_info = rknn_model_info["outputs"]
        for index in range(len(tensors_info)):
            tensor_name = tensors_info[index]["name"]
            tensor_info = self.creat_tensor(engine, tensor_name, "OUTPUT", tensors_info[index]["tensor_info"], log_flag)
            if "url" in tensors_info[index]["tensor_info"].keys():
                tensor_info["alias"] = tensors_info[index]["tensor_info"]["url"]
            engine.add_outputs_info(tensor_name, tensor_info)


    def construct_engine_nodes(self, rknn_model_info:dict, engine:Engine, log_flag=False):
        nodes_info = rknn_model_info["nodes"]
        for index in range(len(nodes_info)):
            node_info = self.constructor.construct_node(nodes_info, index, engine, log_flag)
            engine.add_nodes_info(node_info)


    def construct_engine_norm_info(self, rknn_model_info:dict, engine:Engine, log_flag=False):
        assert "mean_value" in rknn_model_info.keys(), "rknn model info not contain mean value!"
        assert "std_value" in rknn_model_info.keys(), "rknn model info not contain std value!"
        assert "reorder" in rknn_model_info.keys(), "rknn model info not contain reorder!"
        for index in range(len(rknn_model_info["inputs"])):
            item = rknn_model_info["inputs"][index]
            input_name = item["name"]
            engine.set_mean_value(input_name, rknn_model_info["mean_value"])
            engine.set_std_value(input_name, rknn_model_info["std_value"])
            engine.set_reorder(input_name, rknn_model_info["reorder"])
            if log_flag:
                print("engie norm info as follows")
                print("mean value: {}".format(rknn_model_info["mean_value"]))
                print("std  value: {}".format(rknn_model_info["std_value"]))
                print("reorder   : {}".format(rknn_model_info["reorder"]))


    def convert_to_timvx(self, rknn_file:str, log_flag:bool=False):
        assert os.path.isfile(rknn_file), "{} not a valid file path!"
        with open(rknn_file, "rb") as f:
            rknn_model_data = f.read()
        engine = Engine("timvx_engine")
        assert engine.create_graph(), "timvx engine create graph fail!"
        rknn_model_info = parse_rknn_model(rknn_model_data)
        self.format_rknn_model(rknn_model_info)
        self.construct_engine_tensors(rknn_model_info, engine, log_flag)
        self.construct_engine_nodes(rknn_model_info, engine, log_flag)
        self.construct_engine_norm_info(rknn_model_info, engine, log_flag)
        return engine
