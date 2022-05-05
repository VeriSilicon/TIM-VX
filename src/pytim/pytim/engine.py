# -*- coding: utf-8 -*-
import numpy as np
from .lib.timvx import *

class Engine():
    def __init__(self, name):
        self.engine = timvx(name)

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

    def convert_tim_dtype_to_np_dtype(self, datatype):
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

    def get_graph_name(self, ):
        return self.engine.get_graph_name()

    def get_tensor_size(self, tensor_name):
        return self.engine.get_tensor_size(tensor_name)

    def create_tensor(self, tensor_name, tensor_dtype, tensor_attr, tensor_shape, quant_info):
        self.engine.create_tensor(tensor_name, tensor_dtype, tensor_attr, tensor_shape, quant_info)

    def copy_data_from_tensor(self, tensor_name, np_data):
        return self.engine.copy_data_from_tensor(tensor_name, np_data)

    def copy_data_to_tensor(self, tensor_name, np_data):
        return self.engine.copy_data_to_tensor(tensor_name, np_data)

    def create_operation(self, op_info):
        return self.engine.create_operation(op_info)

    def get_op_info(self, op_name):
        self.engine.get_op_info(op_name)

    def bind_input(self, op_name, tensor_name):
        return self.engine.bind_input(op_name, tensor_name)

    def bind_output(self, op_name, tensor_name):
        return self.engine.bind_output(op_name, tensor_name)

    def bind_inputs(self, op_name, tensor_names):
        return self.engine.bind_inputs(op_name, tensor_names)

    def bind_outputs(self, op_name, tensor_names):
        return self.engine.bind_outputs(op_name, tensor_names)

    def set_rounding_policy(self, overflow_policy, rounding_policy,
     down_scale_size_rounding, accumulator_bits):
        pass

    def create_graph(self):
        return self.engine.create_graph()
    
    def compile_graph(self):
        return self.engine.compile_graph()
    
    def run_graph(self):
        return self.engine.run_graph()
 