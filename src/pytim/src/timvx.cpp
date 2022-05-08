/***********************************
******  timvx.cpp
******
******  Created by zhaojd on 2022/04/26.
***********************************/

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <memory>

#include "pybind11/pybind11.h"
#include "timvx_engine.h"

namespace py = pybind11;
using namespace TIMVXPY;

PYBIND11_MODULE(timvx, m)
{
    py::class_<TimVXEngine>(m, "timvx_engine")
    .def(py::init<const std::string &>())
    .def("get_tensor_size",       &TimVXEngine::get_tensor_size)
    .def("create_tensor",         &TimVXEngine::create_tensor)
    .def("copy_data_from_tensor", &TimVXEngine::copy_data_from_tensor)
    .def("copy_data_to_tensor",   &TimVXEngine::copy_data_to_tensor)
    .def("create_operation",      &TimVXEngine::create_operation)
    .def("get_op_info",           &TimVXEngine::get_op_info)
    .def("bind_inputs",           &TimVXEngine::bind_inputs)
    .def("bind_outputs",          &TimVXEngine::bind_outputs)
    .def("bind_input",            &TimVXEngine::bind_input)
    .def("bind_output",           &TimVXEngine::bind_output)
    .def("create_graph",          &TimVXEngine::create_graph)
    .def("compile_graph",         &TimVXEngine::compile_graph)
    .def("run_graph",             &TimVXEngine::run_graph)
    .def("get_graph_name",        &TimVXEngine::get_graph_name);
    // .def("set_rounding_policy",   &TimVXEngine::set_rounding_policy);

    // py::class_<TimVXQuantInfo>(m, "quant_info")
    // .def(py::init<int32_t, int32_t, std::vector<float> &, std::vector<int32_t> &>())
    // .def("get_type",        &TimVXQuantInfo::get_type)
    // .def("get_channel_dim", &TimVXQuantInfo::get_channel_dim)
    // .def("get_scales",      &TimVXQuantInfo::get_scales)
    // .def("get_zero_points", &TimVXQuantInfo::get_zero_points);
}