/***********************************
******  op_creator.h
******
******  Created by zhaojd on 2022/04/27.
***********************************/
#pragma once
#include <map>
#include <iostream>
#include <mutex>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/types.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
using namespace tim::vx;
using namespace std;
namespace py = pybind11;
namespace TIMVXPY
{
    void register_ops();
    class OpCreator 
    {
    public:
        virtual Operation* on_create(std::shared_ptr<Graph> &graph, const py::dict &op_info) = 0;

        bool parse_pool_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, PoolType &pool_type, bool necessary = true);
        bool parse_pad_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, PadType &pad_type, bool necessary = true);
        bool parse_round_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundType &round_type, bool necessary = true);
        bool parse_overflow_policy_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, OverflowPolicy &overflow_policy_type, bool necessary = true);
        bool parse_rounding_policy_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, RoundingPolicy &rounding_policy_type, bool necessary = true);
        bool parse_resize_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, ResizeType &resize_type, bool necessary = true);
        bool parse_data_layout_type(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, DataLayout &data_layout_type, bool necessary = true);
        
        template <class T>
        bool check_obj_type(const py::detail::item_accessor &item)
        {
            return py::isinstance<T>(item);
        }

        template <class T, class NEW_T>
        bool parse_value(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, NEW_T &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << op_name << " op should contain " << attr_name << " attr, please check!" << std::endl;
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
                    std::cout << op_name << " op parse " << attr_name << " attr fail, please check!" << std::endl;
                    return false;
                }
            }
            return true;
        }

        template <class T>
        bool check_list_item_type(const py::list &list_value)
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
        bool parse_fix_list(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, std::array<NEW_T, list_num> &parsed_value, bool necessary = true)
        {
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << op_name << " op should contain " << attr_name << " attr, please check!" << std::endl;
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!check_obj_type<py::list>(op_info[attr_c_name]))
                {
                    std::cout << op_name << " op's attr " << attr_name << " is not list!" << std::endl;
                    return false;
                }
                py::list list_value = py::list(op_info[attr_c_name]);
                if (list_value.size() != list_num)
                {
                    std::cout << op_name << " op's attr " << attr_name << " len should be " << list_num << std::endl;
                    return false;
                }
                if (!check_list_item_type<T>(list_value))
                {
                    std::cout << op_name << " op's attr " << attr_name << " item type wrong!" << std::endl;
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
        bool parse_dynamic_list(const py::dict &op_info, const std::string &op_name, 
            const std::string &attr_name, std::vector<NEW_T> &parsed_value, bool necessary = true)
        {
            parsed_value.clear();
            const char* attr_c_name = attr_name.c_str();
            if (necessary && !op_info.contains(attr_c_name))
            {
                std::cout << op_name << " op should contain " << attr_name << " attr, please check!" << std::endl;
                return false;
            }
            if (op_info.contains(attr_c_name))
            {
                if (!check_obj_type<py::list>(op_info[attr_c_name]))
                {
                    std::cout << op_name << " op's attr " << attr_name << " is not list!" << std::endl;
                    return false;
                }
                py::list list_value = py::list(op_info[attr_c_name]);
                if (!check_list_item_type<T>(list_value))
                {
                    std::cout << op_name << " op's attr " << attr_name << " item type wrong!" << std::endl;
                    return false;
                }
                for (int i = 0; i < list_value.size(); i++)
                {
                    parsed_value.push_back(NEW_T(T(list_value[i])));
                }
            }
            return true;
        }
    };


    class TimVXOp
    {
    private:
        TimVXOp() = default;
    public:
        bool add_creator(std::string op_type, OpCreator* creator);
        OpCreator* get_creator(std::string op_type);
        static TimVXOp* get_instance()
        {
            static TimVXOp instance;
            return &instance;
        }

    private:
        std::map<std::string, OpCreator*> op_creator;
    };


    #define REGISTER_OP_CREATOR(name, op_type)                       \
        void op_type##_op_creator() {                                \
            static name _temp;                                       \
            TimVXOp::get_instance()->add_creator(#op_type, &_temp);  \
        }

}  //namespace TIMVXPY
