/***********************************
******  timvx_engine.h
******
******  Created by zhaojd on 2022/04/25.
***********************************/
#pragma once
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/operation.h"
using namespace tim::vx;
namespace py = pybind11;
namespace TIMVXPY
{
    struct TimVXQuantInfo
    {
        // TimVXQuantInfo(int type, float scale, int32_t zero_point)
        //   : m_type(type), m_scales({scale}), m_zero_points({zero_point}) {}
        TimVXQuantInfo(int32_t type, int32_t channel_dim, std::vector<float> &scales,
               std::vector<int32_t> &zero_points)
            : m_type(type), 
              m_channel_dim(channel_dim), 
              m_scales(scales), 
              m_zero_points(zero_points) {}
        int32_t get_type() {return m_type;}
        int32_t get_channel_dim() {return m_channel_dim;}
        std::vector<float> get_scales() {return m_scales;}
        std::vector<int32_t> get_zero_points() {return m_zero_points;}
        int32_t m_type;
        int32_t m_channel_dim{-1};
        std::vector<float> m_scales;
        std::vector<int32_t> m_zero_points;
    };

    class TimVXEngine 
    {
    public:
        TimVXEngine(const std::string &graph_name);
        ~TimVXEngine();
        // tensor utils
        size_t get_tensor_size(const std::string &tensor_name);
        bool create_tensor(const std::string &tensor_name, const py::dict &tensor_info);
        bool copy_data_from_tensor(const std::string &tensor_name, py::buffer &np_data);
        bool copy_data_to_tensor(const std::string &tensor_name, py::buffer &np_data);
        // operation utils
        bool create_operation(py::dict &op_info);
        py::dict get_op_info(const std::string &op_name);
        bool bind_inputs(const std::string &op_name, const std::vector<std::string> &input_list);
        bool bind_outputs(const std::string &op_name, const std::vector<std::string> &output_list);
        bool bind_input(const std::string &op_name, const std::string &input_name);
        bool bind_output(const std::string &op_name, const std::string &output_name);
        // bool set_rounding_policy(const std::string &op_name, const py::dict &rounding_policy);

        // graph uitls
        bool create_graph();
        bool compile_graph();
        bool run_graph();
        std::string get_graph_name();
    private:
        // util func
        uint32_t type_get_bits(DataType type);
        // operation func
        // tensor names
        std::vector<std::string>                       m_input_tensor_names;
        std::vector<std::string>                       m_output_tensor_names;
        // tensors
        std::map<std::string, std::shared_ptr<Tensor>> m_tensors;
        // std::map<std::string, TensorSpec>              m_tensors_spec;
        std::map<std::string, std::shared_ptr<char>>   m_tensors_data;
        // operation
        std::map<std::string, Operation*>              m_operations;
        std::map<std::string, py::dict>                m_op_info;
        // engine context/graph/name
        std::shared_ptr<Context>                       m_context;
        std::shared_ptr<Graph>                         m_graph;
        std::string                                    m_graph_name;
    };
} //namespace TIMVXPY
