# -*- coding: utf-8 -*-
PadType = ["NONE", "AUTO", "VALID", "SAME"]
PoolType = ["MAX", "AVG", "L2", "AVG_ANDROID"]
RoundType = ["CEILING", "FLOOR"]
OverflowPolicy = ["WRAP", "SATURATE"]
RoundingPolicy = ["TO_ZERO", "RTNE"]
ResizeType = ["NEAREST_NEIGHBOR", "BILINEAR", "AREA"]
DataLayout = [ "ANY", "WHCN", "CWHN", "IcWHOc", "OcIcWH", "IcOcWH", "WHIcOc", "WCN", "WIcOc"]

def ConstructConv2dOpConfig(op_name:str, stride:list, dilation:list, ksize:list=[0, 0], padding:str="AUTO", 
    pad:list=[0, 0, 0, 0], weights:int=0, multiplier:int=0, input_layout:str="WHCN", 
    kernel_layout:str="WHIcOc", op_inputs:list=[], op_outputs:list=[])->dict:

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert input_layout in DataLayout, "input_layout:{} is not in {}".format(input_layout, DataLayout)
    assert kernel_layout in DataLayout, "kernel_layout:{} is not in {}".format(kernel_layout, DataLayout)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Conv2d"
    op_attr = {}
    op_attr["ksize"] = ksize
    op_attr["stride"] = stride
    op_attr["dilation"] = dilation
    op_attr["padding"] = padding
    op_attr["pad"] = pad
    op_attr["weights"] = weights
    op_attr["multiplier"] = multiplier
    op_attr["input_layout"] = input_layout
    op_attr["kernel_layout"] = kernel_layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructActivationOpConfig(op_name:str, activation_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # 1 prelu parameter
    # axis = None
    # 2 leakyrelu parameter
    # ratio = None
    # 3 linear parameter
    # a = None b = 0.0
    # 4 gelu parameter
    # approximate = True    
    valid_act_type = ["Relu", "Relu1", "Relu6", "Elu", "Sigmoid", "Mish", "HardSigmoid",
        "SoftRelu", "HardSwish", "Swish", "Prelu", "Tanh", "LeakyRelu", "Linear", "Gelu"]
    assert activation_type in valid_act_type, "activation_type:{} is not in {}".format(activation_type, valid_act_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Activation"
    op_attr = {}
    op_attr["activation_type"] = activation_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs

    return op_info_dict


def ConstructEltwiseOpConfig(op_name:str, eltwise_type:str, parameter:dict={}, 
    op_inputs:list=[], op_outputs:list=[])->dict:

    # Multiply/Div parameter
    # scale = 1.0
    valid_eltwise_type = ["Minimum", "Maximum", "Add", "Sub", "Pow", "FloorDiv", "Multiply", "Div"]
    assert eltwise_type in valid_eltwise_type, "eltwise_type:{} is not in {}".format(eltwise_type, valid_eltwise_type)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Eltwise"
    op_attr = {}
    op_attr["eltwise_type"] = eltwise_type
    op_attr.update(parameter)
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    
    return op_info_dict

def ConstructReshapeOpConfig(op_name:str, size:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Reshape"
    op_attr = {}
    op_attr["size"] = size
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructTransposeOpConfig(op_name:str, perm:list, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Transpose"
    op_attr = {}
    op_attr["perm"] = perm
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict

def ConstructFullyConnectedOpConfig(op_name:str, axis:int, weights:int=0, op_inputs:list=[], op_outputs:list=[])->dict:

    assert axis >= 0, "axis:{} should >= 0".format(axis)
    assert weights >= 0, "weights:{} should >= 0".format(weights)
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "FullyConnected"
    op_attr = {}
    op_attr["axis"] = axis
    op_attr["weights"] = weights
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructSoftmaxOpConfig(op_name:str, beta:float, axis:int=0, op_inputs:list=[], op_outputs:list=[])->dict:

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Softmax"
    op_attr = {}
    op_attr["beta"] = beta
    op_attr["axis"] = axis
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructResizeOpConfig(op_name:str, type:str, factor:float, align_corners:bool,
        half_pixel_centers:bool, target_height:int, target_width:int, 
        layout:str="WHCN", op_inputs:list=[], op_outputs:list=[]):

    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Resize"
    op_attr = {}
    op_attr["type"] = type
    op_attr["factor"] = factor
    op_attr["align_corners"] = align_corners
    op_attr["half_pixel_centers"] = half_pixel_centers
    op_attr["target_height"] = target_height
    op_attr["target_width"] = target_width
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict


def ConstructPool2dOpConfig(op_name:str, type:str, ksize:list=[], stride:list=[], padding:str="AUTO",
    pad:list=[0, 0, 0, 0], input_size:list=[], output_size:list=[], round_type:str="FLOOR", 
    layout:str="WHCN", op_inputs:list=[], op_outputs:list=[]):

    assert padding in PadType, "padding:{} is not in {}".format(padding, PadType)
    assert round_type in RoundType, "round_type:{} is not in {}".format(round_type, RoundType)
    assert layout in DataLayout, "layout:{} is not in {}".format(layout, DataLayout)
    if len(input_size) == 0:
        assert len(ksize) and len(stride), "ksize and stride len should > 0, when input_size len is 0"
    if len(input_size) > 0:
        assert len(ksize) == 0 and len(stride) == 0, "ksize and stride len should be 0, when input_size len > 0"
    if padding != "AUTO":
        assert pad == [0, 0, 0, 0], "pad should be [0, 0, 0, 0], when padding is not AUTO"
    op_info_dict = {}
    op_info_dict["op_name"] = op_name
    op_info_dict["op_type"] = "Pool2d"
    op_attr = {}
    op_attr["type"] = type
    if len(input_size) > 0 and len(output_size) == 0:
        op_attr["input_size"] = input_size
    elif len(input_size) > 0 and len(output_size) > 0:
        op_attr["input_size"] = input_size
        op_attr["output_size"] = output_size
    elif len(input_size) == 0 and padding == "AUTO":
        op_attr["pad"] = pad
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    else:
        op_attr["padding"] = padding
        op_attr["ksize"] = ksize
        op_attr["stride"] = stride
    op_attr["round_type"] = round_type
    op_attr["layout"] = layout
    op_info_dict["op_attr"] = op_attr
    if len(op_inputs) > 0:
        op_info_dict["op_inputs"] = op_inputs
    if len(op_outputs) > 0:
        op_info_dict["op_outputs"] = op_outputs
    return op_info_dict

