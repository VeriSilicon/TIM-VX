# -*- coding: utf-8 -*-
import numpy as np
from .lib import *

TimVxDataType = ["INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "FLOAT16", "FLOAT32", "BOOL8"]

class Engine():
    def __init__(self, name:str):
        self.engine = timvx_engine(name)
        self.mean_value = {}
        self.std_value = {}
        self.reorder = {}
        self.inputs_info = {}
        self.outputs_info = {}
        self.nodes_info = []
        self.tensors_info = []


    def set_mean_value(self, input_name:str, mean_value:list):
        self.mean_value[input_name] = mean_value


    def set_std_value(self, input_name:str, std_value:list):
        self.std_value[input_name] = std_value


    def set_reorder(self, input_name:str, reorder:list):
        if reorder != [0, 1, 2] and reorder != [2, 1, 0]:
            assert False, "invaid channel reorder {}".format(reorder)
        self.reorder[input_name] = reorder


    def add_inputs_info(self, input_name:str, tensor_info:dict):
        assert input_name not in self.inputs_info, "tensor {} already exists!".format(input_name)
        self.inputs_info[input_name] = tensor_info


    def add_outputs_info(self, output_name:str, tensor_info:dict):
        assert output_name not in self.outputs_info, "tensor {} already exists!".format(output_name)
        self.outputs_info[output_name] = tensor_info


    def add_nodes_info(self, node_info:dict):
        self.nodes_info.append(node_info)


    def add_tensors_info(self, tensor_info:dict):
        self.tensors_info.append(tensor_info)


    def convert_np_dtype_to_tim_dtype(self, datatype):
        if datatype == np.int8:
            return "INT8"
        elif datatype == np.uint8:
            return "UINT8"
        elif datatype == np.int16:
            return "INT16"
        elif datatype == np.uint16:
            return "UINT16"
        elif datatype == np.int32:
            return "INT32"
        elif datatype == np.uint32:
            return "UINT32"
        elif datatype == np.float16:
            return "FLOAT16"
        elif datatype == np.float32:
            return "FLOAT32"
        elif datatype == np.bool:
            return "BOOL8"
        else:
            assert False, "unspoorted datatype {}, when convert np type to tim type".format(datatype)


    def convert_tim_dtype_to_np_dtype(self, datatype:str):
        if datatype == "INT8":
            return np.int8
        elif datatype == "UINT8":
            return np.uint8
        elif datatype == "INT16":
            return np.int16
        elif datatype == "UINT16":
            return np.uint16
        elif datatype == "INT32":
            return np.int32
        elif datatype == "UINT32":
            return np.uint32
        elif datatype == "FLOAT16":
            return np.float16
        elif datatype == "FLOAT32":
            return np.float32
        elif datatype == "BOOL8":
            return np.bool
        else:
            assert False, "unspoorted datatype {}, when convert tim tensor type to np type".format(datatype)


    def get_graph_name(self):
        return self.engine.get_graph_name()


    def get_tensor_size(self, tensor_name:str):
        return self.engine.get_tensor_size(tensor_name)


    def create_tensor(self, tensor_name:str, tensor_dtype:str, tensor_attr:str, \
        tensor_shape:list, quant_info:dict={}, np_data:np.array=np.array([])):

        assert tensor_dtype in TimVxDataType, "tim-vx not support {} datatype".format(tensor_dtype)
        tensor_info = {}
        tensor_info["shape"] = tensor_shape
        tensor_info["data_type"] = tensor_dtype
        tensor_info["attribute"] = tensor_attr
        if len(quant_info.keys()) != 0:
            tensor_info["quant_info"] = quant_info
        if np_data.size != 0:
            tensor_info["data"] = np_data
        return self.engine.create_tensor(tensor_name, tensor_info)


    def copy_data_from_tensor(self, tensor_name:str, np_data:np.array):
        return self.engine.copy_data_from_tensor(tensor_name, np_data)


    def copy_data_to_tensor(self, tensor_name:str, np_data:np.array):
        return self.engine.copy_data_to_tensor(tensor_name, np_data)


    def create_operation(self, op_info:dict):
        ret = self.engine.create_operation(op_info)
        op_name = op_info["op_name"]
        if ret and "op_inputs" in op_info.keys():
            op_inputs = op_info["op_inputs"]
            ret = self.engine.bind_inputs(op_name, op_inputs)
        if ret and "op_outputs" in op_info.keys():
            op_outputs = op_info["op_outputs"]
            ret = self.engine.bind_outputs(op_name, op_outputs)
        return ret


    def get_op_info(self, op_name:str):
        self.engine.get_op_info(op_name)


    def bind_input(self, op_name:str, tensor_name:str):
        return self.engine.bind_input(op_name, tensor_name)


    def bind_output(self, op_name:str, tensor_name:str):
        return self.engine.bind_output(op_name, tensor_name)


    def bind_inputs(self, op_name:str, tensor_names:list):
        return self.engine.bind_inputs(op_name, tensor_names)


    def bind_outputs(self, op_name:str, tensor_names:list):
        return self.engine.bind_outputs(op_name, tensor_names)


    # def set_rounding_policy(self, op_name:str, overflow_policy:str="SATURATE", rounding_policy:str="RTNE",
    #  down_scale_size_rounding:str="FLOOR", accumulator_bits:int=0):
    #     rounding_policy_dict = {}
    #     rounding_policy_dict["overflow_policy"] = overflow_policy
    #     rounding_policy_dict["rounding_policy"] = rounding_policy
    #     rounding_policy_dict["down_scale_size_rounding"] = down_scale_size_rounding
    #     rounding_policy_dict["accumulator_bits"] = accumulator_bits
    #     return self.engine.set_rounding_policy(op_name, rounding_policy_dict)


    def create_graph(self):
        return self.engine.create_graph()


    def compile_graph(self):
        return self.engine.compile_graph()


    def run_graph(self, input_dict:dict):
        for input_name in input_dict.keys():
            assert input_name in self.inputs_info.keys(), "invalid input tensor name {}".format(input_name)
            input_data = input_dict[input_name]
            assert len(input_data.shape) == 3, "need a hwc format input, please check!"
            h,w,c = input_data.shape
            assert c == 3 or c == 1, "input channel should be 1 or 3"
            assert type(input_dict[input_name]) == np.ndarray, "{} tensor data only support numpy array"
            engine_input = (input_data.astype(np.float32) - self.mean_value[input_name]) / self.std_value[input_name]
            if self.reorder == [2, 1, 0]:
                engine_input = engine_input[:,:,::-1]
            engine_input = engine_input.transpose((2, 0, 1))
            shape = self.inputs_info[input_name]["shape"]
            dtype = self.inputs_info[input_name]["dtype"]
            scale = 1.0
            zero_point = 0.0
            if "scale" in self.inputs_info[input_name]["quant_info"]:
                scale = self.inputs_info[input_name]["quant_info"]["scale"]
            if "zero_point" in self.inputs_info[input_name]["quant_info"]:
                zero_point = self.inputs_info[input_name]["quant_info"]["zero_point"]
            engine_input = (engine_input / scale + zero_point).reshape(shape).astype(dtype)
            input_bytes = engine_input.tobytes()
            assert self.engine.copy_data_to_tensor(input_name, input_bytes), "set input {} fail!".format(input_name)

        assert self.engine.run_graph(), "run graph fail!"

        outputs = []
        for output_name in self.outputs_info.keys():
            dtype = self.outputs_info[output_name]["dtype"]
            shape = self.outputs_info[output_name]["shape"]
            scale = 1.0
            zero_point = 0.0
            if "scale" in self.outputs_info[output_name]["quant_info"]:
                scale = self.outputs_info[output_name]["quant_info"]["scale"]
            if "zero_point" in self.outputs_info[output_name]["quant_info"]:
                zero_point = self.outputs_info[output_name]["quant_info"]["zero_point"]
            output_data = np.zeros(shape).astype(dtype)
            assert self.engine.copy_data_from_tensor(output_name, output_data), "get output {} fail!".format(output_name)
            output_data = output_data.astype(np.float32)
            output_data = (output_data - zero_point) * scale
            output_data = output_data.reshape(shape[::-1])
            outputs.append(output_data)
        return outputs
