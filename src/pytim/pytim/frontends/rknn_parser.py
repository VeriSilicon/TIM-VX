# -*- coding: utf-8 -*-
"""rknn frontend."""
import json

def get_model_signature(rknn_data, offset):
    signature_len = 8
    start = offset
    end = offset + signature_len
    signature = rknn_data[start:end]
    if signature[0:4].decode() == "RKNN":
        signature_str = "RKNN"
    elif signature[0:8].decode() == "CYPTRKNN":
        signature_str = "CYPTRKNN"
    else:
        assert False, "wrong rknn signature"
    return signature_str, offset + signature_len


def get_model_version(rknn_data, offset):
    version_len = 8
    start = offset
    end = offset + version_len
    version_data = rknn_data[start:end]
    version = int.from_bytes(version_data, byteorder='little', signed=False)
    return version, offset + version_len


def get_weights_size(rknn_data, offset):
    weight_size_len = 8
    start = offset
    end = offset + weight_size_len
    weight_size_data = rknn_data[start:end]
    weight_size = int.from_bytes(weight_size_data, byteorder='little', signed=False)
    return weight_size, offset + weight_size_len


def get_model_size(rknn_data, offset):
    model_size_len = 8
    start = offset
    end = offset + model_size_len
    model_size_data = rknn_data[start:end]
    model_size = int.from_bytes(model_size_data, byteorder='little', signed=False)
    return model_size, offset + model_size_len


def parse_model_nodes(nodes_info_json_list, tensor_info):
    nodes_info = []
    for item_info in nodes_info_json_list:
        node_info = {}
        if "name" in item_info.keys():
            name = item_info["name"]
        else:
            name = ""
        if 'VSI_NN_OP_' in item_info["op"]:
            prefix_len = len('VSI_NN_OP_')
            type = item_info["op"][prefix_len:]
        elif 'RKNN_OP_' in item_info["op"]:
            prefix_len = len('RKNN_OP_')
            type = item_info["op"][prefix_len:]
        else:
            assert False, "invalid op type: {}".format(item_info["op"])
        inputs = []
        outputs = []
        for index in range(len(item_info["input"])):
            input_tensor_info = item_info["input"][index]
            if "right_tensor" in input_tensor_info.keys():
                tensor_name = str(input_tensor_info["right_tensor"]["type"]) + ':' + str(input_tensor_info["right_tensor"]["tensor_id"])
                inputs.append(tensor_name)
            elif "right_node" in input_tensor_info.keys():
                tensor_name = str(input_tensor_info["right_node"]["node_id"]) + ':' + str(input_tensor_info["right_node"]["tensor_id"])
                inputs.append(tensor_name)
            else:
                print("{} input {} have no right_tensor or right_node".format(name, index))

        for index in range(len(item_info["output"])):
            output_tensor_info = item_info["output"][index]
            if "right_tensor" in output_tensor_info.keys():
                tensor_name = str(output_tensor_info["right_tensor"]["type"]) + ':' + str(output_tensor_info["right_tensor"]["tensor_id"])
                outputs.append(tensor_name)
            elif "right_node" in output_tensor_info.keys():
                tensor_name = str(output_tensor_info["right_node"]["node_id"]) + ':' + str(output_tensor_info["right_node"]["tensor_id"])
                outputs.append(tensor_name)
            else:
                print("{} output {} have no right_tensor or right_node".format(name, index))
        attribute = {}
        if "nn" in item_info.keys():
            for item_key in item_info["nn"].keys():
                attribute[item_key] = item_info["nn"][item_key]
        if "vx" in item_info.keys():
                attribute["vx"] = item_info["vx"]
        
        node_info["name"] = name
        node_info["type"] = type
        node_info["inputs"] = inputs
        node_info["outputs"] = outputs
        node_info["attribute"] = attribute
        nodes_info.append(node_info)
    return nodes_info

def parse_const_tensor(const_tensor_list, weights_data, tensor_info):
    for const_tensor in const_tensor_list:
        item_info = {}
        tensor_name = 'const_tensor:' + str(const_tensor["tensor_id"])
        item_info["shape"] = const_tensor["size"]
        item_info["dtype"] = const_tensor["dtype"]
        offset = const_tensor["offset"]
        weight_len = const_tensor["len"]
        item_info["raw_data"] = weights_data[offset: offset+weight_len]
        tensor_info[tensor_name] = item_info


def parse_virtual_tensor(virtual_tensor_list, tensor_info):
    for virtual_tensor in virtual_tensor_list:
        item_info = {}
        tensor_name = str(virtual_tensor["node_id"]) + ":" + str(virtual_tensor["output_port"])
        item_info["dtype"] = virtual_tensor["dtype"]
        tensor_info[tensor_name] = item_info


def parse_norm_tensor(norm_tensor_list, tensor_info):
    for norm_tensor in norm_tensor_list:
        item_info = {}
        tensor_name = 'norm_tensor:' + str(norm_tensor["tensor_id"])
        item_info["shape"] = norm_tensor["size"]
        item_info["dtype"] = norm_tensor["dtype"]
        item_info["url"] = norm_tensor["url"]
        tensor_info[tensor_name] = item_info


def init_model_nodes(nodes_list):
    for node in nodes_list:
        node["input"] = []
        node["output"] = []


def parse_model_connection(connection_list, nodes_list):
    for connection in connection_list:
        if connection["left"] == "input":
            node_id = connection["node_id"]
            nodes_list[node_id]["input"].append(connection)
            if "right_node" in connection.keys():
                node_id = connection["right_node"]["node_id"]
                tensor_id = connection["right_node"]["tensor_id"]
                if tensor_id >= len(nodes_list[node_id]["output"]):
                    num = len(nodes_list[node_id]["output"]) - tensor_id + 1
                    nodes_list[node_id]["output"].extend([None] * num)
                nodes_list[node_id]["output"][tensor_id] = connection
        elif connection["left"] == "output":
            node_id = connection["node_id"]
            nodes_list[node_id]["output"].append(connection)
        else:
            assert False, "unspported connection {}".format(connection)


def parse_model_graph(graph_info_list, tensor_info):
    inputs = []
    outputs = []
    for item_info in graph_info_list:
        key = item_info["right"] + ':' + str(item_info["right_tensor_id"])
        tensor = tensor_info[key]
        # if item_info["left_tensor_id"] == 0:
        #     name = item_info["left"]
        # else:
        #     name = item_info["left"] + str(item_info["left_tensor_id"])
        parameter = {}
        parameter["name"] = key
        parameter["tensor_info"] = tensor
        if item_info["left"] == "input":
            inputs.append(parameter)
        elif item_info["left"] == "output":
            outputs.append(parameter)
        else:
            assert False, "unsupport graph {}".format(item_info)

    return inputs, outputs


# def check_model_info(model_info):
#     assert model_info["input_num"] == len(model_info["inputs"]), "model input_number{} not equal parsed input_number{}".format(model_info["input_num"], len(model_info["inputs"]))
#     assert model_info["output_num"] == len(model_info["outputs"]), "model output_number{} not equal parsed output_number{}".format(model_info["output_num"], len(model_info["outputs"]))
#     for tensor_key in model_info["tensors"].keys():
#         tensor = model_info["tensors"][tensor_key]
#         if tensor['dtype']['qnt_type'] != 'VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC' or \
#             (tensor['dtype']['vx_type'] != 'VSI_NN_TYPE_UINT8' and tensor['dtype']['vx_type'] != 'VSI_NN_TYPE_INT32'):
#             print("{} qnt_type is {}, vx_type is {}".format(tensor_key, tensor['dtype']['qnt_type'], tensor['dtype']['vx_type']))
#             assert False, "current noly support VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC and VSI_NN_TYPE_UINT8"


def parse_model_json(model_json, weights_data):
    model_info = {}
    model_info["target_platform"] = model_json["target_platform"]
    model_info["network_platform"] = model_json["network_platform"]
    model_info["ori_network_platform"] = model_json["ori_network_platform"]
    model_info["input_fmt"] = model_json["input_fmt"]
    model_info["input_transpose"] = model_json["input_transpose"]
    model_info["name"] = model_json["name"]
    model_info["version"] = model_json["version"]
    model_info["ovxlib_version"] = model_json["ovxlib_version"]
    model_info["node_num"] = model_json["node_num"]
    model_info["norm_tensor_num"] = model_json["norm_tensor_num"]
    model_info["const_tensor_num"] = model_json["const_tensor_num"]
    model_info["virtual_tensor_num"] = model_json["virtual_tensor_num"]
    model_info["optimization_level"] = model_json["optimization_level"]
    model_info["mean_value"] = model_json["mean_value"]
    model_info["mean_value_chns"] = model_json["mean_value_chns"]
    model_info["std_value"] = model_json["std_value"]
    model_info["reorder"] = model_json["reorder"]
    model_info["input_num"] = model_json["input_num"]
    model_info["output_num"] = model_json["output_num"]
    model_info["pre_compile"] = model_json["pre_compile"]
    model_info["case_type"] = model_json["case_type"]
    tensor_info = {}
    parse_const_tensor(model_json["const_tensor"], weights_data, tensor_info)
    parse_virtual_tensor(model_json["virtual_tensor"], tensor_info)
    parse_norm_tensor(model_json["norm_tensor"], tensor_info)
    init_model_nodes(model_json["nodes"])
    parse_model_connection(model_json["connection"], model_json["nodes"])
    model_inputs, model_outputs = parse_model_graph(model_json["graph"], tensor_info)
    model_nodes = parse_model_nodes(model_json["nodes"], tensor_info)
    model_info["inputs"] = model_inputs
    model_info["outputs"] = model_outputs
    model_info["nodes"] = model_nodes
    model_info["tensors"] = tensor_info
    return model_info

def parse_rknn_model(rknn_data):
    offset = 0
    signature, offset = get_model_signature(rknn_data, offset)
    print(signature)
    if signature == 'CYPTRKNN':
        assert False, "not support cypt rknn model"
    
    version, offset = get_model_version(rknn_data, offset)
    weight_size, offset = get_weights_size(rknn_data, offset)
    print("version: {}".format(version))
    print("weight size: {}".format(weight_size))
    if version > 1:
        offset += 40
    weights_data = rknn_data[offset: offset+weight_size]
    offset += weight_size

    model_size, offset = get_model_size(rknn_data, offset)
    print("model size: {}".format(model_size))
    if offset + model_size != len(rknn_data):
        assert False, "rknn model file is not valid"
    model_data = rknn_data[offset: offset+model_size]
    model_json = json.loads(str(model_data, 'utf-8'))
    # with open("model.json", "w") as f:
    #     json.dump(model_json, f)
    model_info = parse_model_json(model_json, weights_data)
    # check_model_info(model_info)
    return model_info
