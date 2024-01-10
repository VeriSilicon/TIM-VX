/****************************************************************************
*
*    Copyright (c) 2020-2024 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <array>
#include <filesystem>
#include <string_view>

#include "vx/ovx_executor.hpp"
#include "vx/utils.hpp"

namespace vsi::nbg_runner::python {
namespace py = pybind11;
namespace fs = std::filesystem;

PYBIND11_MODULE(_nbg_runner, m) {
  using namespace vsi::nbg_runner::vx;

  // clang-format off
  py::class_<OVXExecutor>(m, "OVXExecutor")
    .def(py::init<const fs::path&>())
    .def(py::init([](const py::buffer& nbg_buffer) {
      auto buffer_info = nbg_buffer.request(false);
      return std::make_unique<OVXExecutor>(reinterpret_cast<char*>(buffer_info.ptr), buffer_info.size);
    }))
    .def("init", &OVXExecutor::init)
    .def("get_num_inputs", &OVXExecutor::get_num_inputs)
    .def("get_num_outputs", &OVXExecutor::get_num_outputs)
    .def("get_input_info", &OVXExecutor::get_input_info)
    .def("get_output_info", &OVXExecutor::get_output_info)
    .def("set_input", [](OVXExecutor* executor, size_t index, const py::buffer& buffer) {
      auto buffer_info = buffer.request(false);
      std::array<size_t, OVXTensorInfo::kMaxRank> vx_shape = {0};
      std::array<size_t, OVXTensorInfo::kMaxRank> vx_strides = {0};
      std::reverse_copy(buffer_info.shape.cbegin(), buffer_info.shape.cend(), vx_shape.begin());
      std::reverse_copy(buffer_info.strides.cbegin(), buffer_info.strides.cend(), vx_strides.begin());
      executor->copy_to_input(
        index,
        buffer_info.ptr,
        buffer_info.ndim,
        vx_shape.data(),
        vx_strides.data()
      );
    })
    .def("get_output", [](OVXExecutor* executor, size_t index) -> py::array {
      auto tensor_info = executor->get_output_info(index);
      auto np_dtype = py::dtype(get_vx_dtype_str(tensor_info.data_type).data());
      auto np_shape = std::vector<ssize_t>(tensor_info.rank);
      std::reverse_copy(tensor_info.shape.data(), tensor_info.shape.data() + tensor_info.rank, np_shape.begin());

      auto np_tensor = py::array(np_dtype, np_shape);
      auto buffer_info = np_tensor.request(true);

      std::array<size_t, OVXTensorInfo::kMaxRank> vx_strides = {0};
      std::reverse_copy(buffer_info.strides.cbegin(), buffer_info.strides.cend(), vx_strides.begin());

      executor->copy_from_output(
        index,
        buffer_info.ptr,
        tensor_info.rank,
        tensor_info.shape.data(),
        vx_strides.data()
      );
      return np_tensor;
    })
    .def("run", &OVXExecutor::run)
  ;

  py::class_<OVXTensorInfo>(m, "OVXTensorInfo")
    .def_readonly("rank", &OVXTensorInfo::rank)
    .def_property_readonly("shape", [](OVXTensorInfo* tensor_info) -> py::tuple {
      size_t rank = tensor_info->rank;
      auto shape_tuple = py::tuple(rank); 
      for (size_t i = 0; i < rank; i++){
        shape_tuple[i] = tensor_info->shape[rank - i - 1];
      }
      return shape_tuple;
    })
    .def_property_readonly("dtype", [](OVXTensorInfo* tensor_info) -> std::string_view {
      return get_vx_dtype_str(tensor_info->data_type);
    })
    .def_property_readonly("qtype", [](OVXTensorInfo* tensor_info) -> std::string_view {
      return get_vx_qtype_str(tensor_info->quant_type);
    })
    .def_property_readonly("scale", [](OVXTensorInfo* tensor_info) -> float {
      return tensor_info->quant_type == VX_QUANT_AFFINE_SCALE
        ? tensor_info->quant_param.affine.scale
        : 0.0F;
    })
    .def_property_readonly("zero_point", [](OVXTensorInfo* tensor_info) -> int32_t {
      return tensor_info->quant_type == VX_QUANT_AFFINE_SCALE
        ? tensor_info->quant_param.affine.zeroPoint
        : 0;
    })
     .def_property_readonly("fixed_point_pos", [](OVXTensorInfo* tensor_info) -> int8_t {
      return tensor_info->quant_type == VX_QUANT_DYNAMIC_FIXED_POINT
        ? tensor_info->quant_param.dfp.fixed_point_pos
        : static_cast<int8_t>(0);
    })
  ;
  // clang-format on
}

}  // namespace vsi::nbg_runner::python
