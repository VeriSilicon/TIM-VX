/***********************************
******  tensor_info.cpp
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#include "tensor_info.h"
#include <iostream>
#include <map>
using namespace tim::vx;
using namespace std;
namespace py = pybind11;

namespace TIMVXPY
{

    bool TensorSpecConstruct::parse_tensor_data_type(const py::dict &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, DataType &data_type)
    {
        std::string data_type_str;
        std::map<std::string, DataType> data_type_map;
        data_type_map["INT8"]     = DataType::INT8;
        data_type_map["UINT8"]    = DataType::UINT8;
        data_type_map["INT16"]    = DataType::INT16;
        data_type_map["UINT16"]   = DataType::UINT16;
        data_type_map["INT32"]    = DataType::INT32;
        data_type_map["UINT32"]   = DataType::UINT32;
        data_type_map["FLOAT16"]  = DataType::FLOAT16;
        data_type_map["FLOAT32"]  = DataType::FLOAT32;
        data_type_map["BOOL8"]    = DataType::BOOL8;
        bool parse_result = parse_value<py::str, std::string>(tensor_info, tensor_name, 
            key_name, data_type_str);
        if (parse_result)
        {
            if (data_type_map.find(data_type_str) != data_type_map.end())
                data_type = data_type_map[data_type_str];
            else
            {
                std::cout << "tensor " << tensor_name << " attr " << key_name 
                    << " contains invalid data type " << data_type_str << "!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }


    bool TensorSpecConstruct::parse_tensor_attr(const py::dict &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, TensorAttribute &tensor_attr)
    {
        std::string tensor_attr_str;
        std::map<std::string, TensorAttribute> tensor_attr_map;
        tensor_attr_map["CONSTANT"]     = TensorAttribute::CONSTANT;
        tensor_attr_map["TRANSIENT"]    = TensorAttribute::TRANSIENT;
        tensor_attr_map["VARIABLE"]     = TensorAttribute::VARIABLE;
        tensor_attr_map["INPUT"]        = TensorAttribute::INPUT;
        tensor_attr_map["OUTPUT"]       = TensorAttribute::OUTPUT;
        bool parse_result = parse_value<py::str, std::string>(tensor_info, tensor_name, 
            key_name, tensor_attr_str);
        if (parse_result)
        {
            if (tensor_attr_map.find(tensor_attr_str) != tensor_attr_map.end())
                tensor_attr = tensor_attr_map[tensor_attr_str];
            else
            {
                std::cout << "tensor " << tensor_name << " attr " << key_name 
                    << " contains invalid attribute type " << tensor_attr_str << "!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }


    bool TensorSpecConstruct::parse_tensor_quant_type(const py::dict &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, QuantType &quant_type)
    {
        std::string quant_type_str;
        std::map<std::string, QuantType> quant_type_map;
        quant_type_map["NONE"]                      = QuantType::NONE;
        quant_type_map["ASYMMETRIC"]                = QuantType::ASYMMETRIC;
        quant_type_map["SYMMETRIC_PER_CHANNEL"]     = QuantType::SYMMETRIC_PER_CHANNEL;
        bool parse_result = parse_value<py::str, std::string>(tensor_info, tensor_name, key_name, quant_type_str);
        if (parse_result)
        {
            if (quant_type_map.find(quant_type_str) != quant_type_map.end())
                quant_type = quant_type_map[quant_type_str];
            else
            {
                std::cout << "tensor " << tensor_name << " attr " << key_name 
                    << " contains invalid quant type " << quant_type_str << "!" << std::endl;
                parse_result = false;
            }
        }
        return parse_result;
    }

    bool TensorSpecConstruct::construct_tensorspec(const py::dict &tensor_info, 
        const std::string &tensor_name, TensorSpec& tensorspec)
    {
        float scale = 1.0;
        int32_t zero_point = 0;
        int channel_dim = -1;
        std::vector<float> scales;
        std::vector<int32_t> zero_points;
        std::vector<uint32_t> shape;
        QuantType quant_type;
        TensorAttribute tensor_attr;
        DataType data_type;
        if (!parse_dynamic_list<py::int_, uint32_t>(tensor_info, tensor_name, "shape", shape)
            || !parse_tensor_attr(tensor_info, tensor_name, "attribute", tensor_attr)
            || !parse_tensor_data_type(tensor_info, tensor_name, "data_type", data_type))
            return false;
        if (tensor_info.contains("quant_info"))
        {
            if (!check_obj_type<py::dict>(tensor_info["quant_info"]))
            {
                std::cout << tensor_name << "'s quant_info should be a dict item!" << std::endl;
                return false;
            }
            py::dict quant_info = py::cast<py::dict>(tensor_info["quant_info"]);
            if (!parse_tensor_quant_type(quant_info, tensor_name, "quant_type", quant_type))
                return false;
            if (quant_info.contains("channel_dim"))
            {
                if (!parse_value<py::int_, int>(quant_info, tensor_name, "channel_dim", channel_dim))
                    return false;
                if (channel_dim < 0)
                {
                    std::cout << tensor_name << "'s channel dim should greater than 0!" << std::endl;
                    return false;
                }
                if (!parse_dynamic_list<py::int_, int32_t>(quant_info, tensor_name, "zero_points", zero_points)
                    || !parse_dynamic_list<py::float_, float>(quant_info, tensor_name, "scales", scales))
                    return false;
                if (zero_points.size() != channel_dim || scales.size() != channel_dim)
                {
                    std::cout << tensor_name << "'s zero_points/scales len is not equal to channel dim!" << std::endl;
                    return false;
                }
            }
            else
            {
                if (!parse_value<py::float_, float>(quant_info, tensor_name, "scale", scale, false)
                    || !parse_value<py::int_, int32_t>(quant_info, tensor_name, "zero_point", zero_point, false))
                    return false;
            }
            if (channel_dim < 0)
            {
                channel_dim = 1;
                scales.clear();
                zero_points.clear();
                scales.push_back(scale);
                zero_points.push_back(zero_point);
            }
            Quantization input_quant(quant_type, channel_dim, scales, zero_points);
            TensorSpec temp_tensor_spec(data_type, shape, tensor_attr, input_quant);
            tensorspec = temp_tensor_spec;
        }
        else
        {
            TensorSpec temp_tensor_spec(data_type, shape, tensor_attr);
            tensorspec = temp_tensor_spec;
        }
        return true;
    }
}