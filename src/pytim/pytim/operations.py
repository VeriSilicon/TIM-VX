# -*- coding: utf-8 -*-
import abc

PadType = ["auto", "valid", "same"]
PoolType = ["max", "avg", "l2", "avg_android"]
RoundType = ["ceiling", "floor"]
OverflowPolicy = ["wrap", "saturate"]
RoundingPolicy = ["to_zero", "rtne"]
ResizeType = ["nearest_neighbor", "bilinear", "area"]
DataLayout = [ "any", "whcn", "cwhn", "icwhoc", "ocicwh", "icocwh", "whicoc", "wcn", "wicoc"]

class OperationConfig(abc.ABCMeta):
    op_name = ""
    op_type = ""
    op_inputs = []
    op_outputs = []
    @abc.abstractmethod
    def get_op_info(self)->dict:
        pass

    def set_inputs(self, tensor_names:list):
        self.op_inputs = tensor_names

    def set_outputs(self, tensor_names:list):
        self.op_outputs = tensor_names

class Conv2dOperationConfig(OperationConfig):
    def __init__(self, op_name:str, stride:list, dilation:list, ksize:list=[0, 0], padding:str="auto", 
        pad:list=[0, 0, 0, 0], weights:int=0, multiplier:int=0, 
        input_layout:str="whcn", kernel_layout:str="whicoc"):
        
        self.op_name = op_name
        self.op_type = "Conv2d"
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.pad = pad
        self.weights = weights
        self.multiplier = multiplier
        self.input_layout = input_layout
        self.kernel_layout = kernel_layout

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["ksize"] = self.ksize
        op_attr["stride"] = self.stride
        op_attr["dilation"] = self.dilation
        op_attr["padding"] = self.padding
        op_attr["pad"] = self.pad
        op_attr["weights"] = self.weights
        op_attr["multiplier"] = self.multiplier
        op_attr["input_layout"] = self.input_layout
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict

class ActivationOperationConfig(OperationConfig):
    activation_type = ""
    activation_para = {}
    # prelu parameter
    # axis = None
    # leakyrelu parameter
    # ratio = None
    # linear parameter
    # a = None
    # b = 0.0
    # gelu parameter
    # approximate = True

    def __init__(self, op_name:str, activation_type:str, parameter:dict={}):
        self.op_name = op_name
        self.op_type = "Activation"
        self.activation_type = activation_type
        self.activation_para = parameter

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["activation_type"] = self.activation_type
        op_attr.update(self.activation_para)
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict


class EltwiseOperationConfig(OperationConfig):
    eltwise_type = ""
    eltwise_para = {}
    # Multiply/Div parameter
    # scale = 1.0

    def __init__(self, op_name:str, eltwise_type:str, parameter:dict={}):
        self.op_name = op_name
        self.op_type = "Eltwise"
        self.eltwise_type = eltwise_type
        self.eltwise_para = parameter

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["eltwise_type"] = self.eltwise_type
        op_attr.update(self.eltwise_para)
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict

class ReshapeOperationConfig(OperationConfig):
    def __init__(self, op_name:str, size:list):
        self.op_name = op_name
        self.op_type = "Reshape"
        self.size = size

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["size"] = self.size
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict


class TransposeOperationConfig(OperationConfig):
    def __init__(self, op_name:str, perm:list):
        self.op_name = op_name
        self.op_type = "Reshape"
        self.perm = perm

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["perm"] = self.perm
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict

class ResizeOperationConfig(OperationConfig):
    def __init__(self, op_name:str, type:str, factor:float, align_corners:bool,
        half_pixel_centers:bool, target_height:int, target_width:int, layout:str="whcn"):
        self.op_name = op_name
        self.op_type = "Resize"
        self.type = type
        self.factor = factor
        self.align_corners = align_corners
        self.half_pixel_centers = half_pixel_centers
        self.target_height = target_height
        self.target_width = target_width
        self.layout = layout

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["type"] = self.type
        op_attr["factor"] = self.factor
        op_attr["align_corners"] = self.align_corners
        op_attr["half_pixel_centers"] = self.half_pixel_centers
        op_attr["target_height"] = self.target_height
        op_attr["target_width"] = self.target_width
        op_attr["layout"] = self.layout
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict


class Pool2dOperationConfig(OperationConfig):
    def __init__(self, op_name:str, type:str, ksize:list, stride:list, padding:str="auto",
        pad:list=[0, 0, 0, 0], input_size:list=[], output_size:list=[], round_type:str="floor", layout:str="whcn"):
        self.op_name = op_name
        self.op_type = "Pool2d"
        self.type = type
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.pad = pad
        self.input_size = input_size
        self.output_size = output_size
        self.round_type = round_type
        self.layout = layout

    def get_op_info(self)->dict:
        op_info_dict = {}
        op_info_dict["op_name"] = self.op_name
        op_info_dict["op_type"] = self.op_type
        op_attr = {}
        op_attr["type"] = self.type
        op_attr["factor"] = self.factor
        op_attr["align_corners"] = self.align_corners
        op_attr["half_pixel_centers"] = self.half_pixel_centers
        op_attr["target_height"] = self.target_height
        op_attr["target_width"] = self.target_width
        op_attr["layout"] = self.layout
        op_info_dict["op_attr"] = op_attr
        op_info_dict["op_inputs"] = self.op_inputs
        op_info_dict["op_outputs"] = self.op_outputs
        return op_info_dict

