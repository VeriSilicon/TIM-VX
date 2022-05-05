/***********************************
******  tensor_info.h
******
******  Created by zhaojd on 2022/05/04.
***********************************/
#pragma once
#include "pybind11/pybind11.h"
#include "tim/vx/tensor.h"
#include "tim/vx/types.h"
using namespace tim::vx;
using namespace std;
namespace py = pybind11;
namespace TIMVXPY
{

    class TensorSpecConstruct
    {
    public:
        static bool construct_tensorspec(const py::dict &tensor_info, const std::string &tensor_name, 
            TensorSpec& tensorspec);

    private:
        static bool parse_tensor_data_type(const py::dict &tensor_info, const std::string &tensor_name, 
            const std::string &key_name, DataType &data_type);
        static bool parse_tensor_attr(const py::dict &tensor_info, const std::string &tensor_name, 
            const std::string &key_name, TensorAttribute &tensor_attr);
        static bool parse_tensor_quant_type(const py::dict &tensor_info, const std::string &tensor_name, 
        const std::string &key_name, QuantType &quant_type);

        template <class T>
        static bool check_obj_type(const py::detail::item_accessor &item)
        {
            return py::isinstance<T>(item);
        }

        template <class T>
        static bool check_list_item_type(const py::list &list_value)
        {
            bool ret = true;
            for (int i = 0; i < list_value.size(); i++)
            {
                ret = py::isinstance<T>(list_value[i]);
                if (false == ret)
                    break;
            }
            return ret;
        }

        template <class T, class NEW_T, int list_num>
        static bool parse_fix_list(const py::dict &op_info, const std::string &tensor_name, 
            const std::string &attr_name, std::array<NEW_T, list_num> parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << tensor_name << " tensor should contain " << attr_name << " attr, please check!" << std::endl;
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!check_obj_type<py::list>(op_info[attr_c_name]))
                {
                    std::cout << tensor_name << " tensor's attr " << attr_name << " is not list!" << std::endl;
                    return false;
                }
                py::list list_value = py::list(op_info[attr_c_name]);
                if (list_value.size() != list_num)
                {
                    std::cout << tensor_name << " tensor's attr " << attr_name << " len should be " << list_num << std::endl;
                    return false;
                }
                if (!check_list_item_type<T>(list_value))
                {
                    std::cout << tensor_name << " tensor's attr " << attr_name << " item type wrong!" << std::endl;
                    return false;
                }
                for (int i = 0; i < list_value.size(); i++)
                {
                    parsed_value[i] = NEW_T(T(list_value[i]));
                }
            }
            return true;
        }

        template <class T, class NEW_T>
        static bool parse_dynamic_list(const py::dict &op_info, const std::string &tensor_name, 
            const std::string &attr_name, std::vector<NEW_T> &parsed_value, bool necessary = true)
        {
            parsed_value.clear();
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << tensor_name << " tensor should contain " << attr_name << " attr, please check!" << std::endl;
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!check_obj_type<py::list>(op_info[attr_c_name]))
                {
                    std::cout << tensor_name << " tensor's attr " << attr_name << " is not list!" << std::endl;
                    return false;
                }
                py::list list_value = py::list(op_info[attr_c_name]);
                if (!check_list_item_type<T>(list_value))
                {
                    std::cout << tensor_name << " tensor's attr " << attr_name << " item type wrong!" << std::endl;
                    return false;
                }
                for (int i = 0; i < list_value.size(); i++)
                {
                    parsed_value.push_back(NEW_T(T(list_value[i])));
                }
            }
            return true;
        }

        template <class T, class NEW_T>
        static bool parse_value(const py::dict &op_info, const std::string &tensor_name, 
            const std::string &attr_name, NEW_T &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << tensor_name << " tensor should contain " << attr_name << " attr, please check!" << std::endl;
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (check_obj_type<T>(op_info[attr_c_name]))
                {
                    parsed_value = NEW_T(T(op_info[attr_c_name]));
                }
                else
                {
                    std::cout << tensor_name << " tensor parse " << attr_name << " attr fail, please check!" << std::endl;
                    return false;
                }
            }
            return true;
        }
    };

} // namespace TIMVXPY

