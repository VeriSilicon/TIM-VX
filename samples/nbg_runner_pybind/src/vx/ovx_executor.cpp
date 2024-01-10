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

#include "ovx_executor.hpp"

#include <VX/vx_khr_import_kernel.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>

#include "utils.hpp"

namespace vsi::nbg_runner::vx {

OVXExecutor::OVXExecutor(const char* nbg_data, size_t nbg_size) {
  nbg_buffer_ = std::vector<char>(nbg_data, nbg_data + nbg_size);
}

OVXExecutor::OVXExecutor(const fs::path& nbg_path) {
  size_t nbg_size = fs::file_size(nbg_path);
  auto nbg_file = std::ifstream(nbg_path, std::ios::in | std::ios::binary);

  nbg_buffer_.resize(nbg_size);
  nbg_file.read(nbg_buffer_.data(), static_cast<std::streamsize>(nbg_size));
}

OVXExecutor::~OVXExecutor() {
  for (auto& tensor : input_tensors_) {
    vxReleaseTensor(&tensor);
  }
  for (auto& tensor : output_tensors_) {
    vxReleaseTensor(&tensor);
  }

  vxReleaseNode(&nbg_node_);
  vxReleaseKernel(&nbg_kernel_);
  vxReleaseGraph(&graph_);
  vxReleaseContext(&context_);
}

int OVXExecutor::init() {
  vx_status status = VX_SUCCESS;

  context_ = vxCreateContext();
  if (context_ == nullptr) {
    throw std::runtime_error("Failed to create OpenVX context.");
  }

  graph_ = vxCreateGraph(context_);
  status = vxGetStatus(reinterpret_cast<vx_reference>(graph_));
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to create OpenVX graph.");
  }

  nbg_kernel_ = vxImportKernelFromURL(
      context_, VX_VIVANTE_IMPORT_KERNEL_FROM_POINTER, nbg_buffer_.data());
  status = vxGetStatus(reinterpret_cast<vx_reference>(nbg_kernel_));
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to import NBG kernel.");
  }

  status = query_nbg_io_infos();
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to query NBG I/O params.");
  }
  size_t num_inputs = input_tensors_infos_.size();
  size_t num_outputs = output_tensors_infos_.size();

  nbg_node_ = vxCreateGenericNode(graph_, nbg_kernel_);
  status = vxGetStatus(reinterpret_cast<vx_reference>(nbg_node_));
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to create NBG node.");
  }

  // Create input tensors and bind to NBG node.
  for (size_t i = 0; i < num_inputs; i++) {
    const auto& tensor_info = input_tensors_infos_[i];
    std::array<uint32_t, OVXTensorInfo::kMaxRank> shape;
    std::transform(tensor_info.shape.cbegin(), tensor_info.shape.cend(),
                   shape.begin(),
                   [](size_t s) { return static_cast<uint32_t>(s); });

    vx_tensor_create_params_t tensor_create_params = {
        .num_of_dims = static_cast<uint32_t>(tensor_info.rank),
        .sizes = shape.data(),
        .data_format = tensor_info.data_type,
        .quant_format = tensor_info.quant_type,
        .quant_data = tensor_info.quant_param,
    };
    vx_tensor input_tensor = vxCreateTensor2(context_, &tensor_create_params,
                                             sizeof(tensor_create_params));
    if (input_tensor == nullptr) {
      throw std::runtime_error("Failed to create input vx tensor.");
    }

    vxSetParameterByIndex(nbg_node_, i,
                          reinterpret_cast<vx_reference>(input_tensor));
    input_tensors_.push_back(input_tensor);
  }

  // Create output tensors and bind to NBG node.
  for (size_t i = 0; i < num_outputs; i++) {
    const auto& tensor_info = output_tensors_infos_[i];
    std::array<uint32_t, OVXTensorInfo::kMaxRank> shape;
    std::transform(tensor_info.shape.cbegin(), tensor_info.shape.cend(),
                   shape.begin(),
                   [](size_t s) { return static_cast<uint32_t>(s); });

    vx_tensor_create_params_t tensor_create_params = {
        .num_of_dims = static_cast<uint32_t>(tensor_info.rank),
        .sizes = shape.data(),
        .data_format = tensor_info.data_type,
        .quant_format = tensor_info.quant_type,
        .quant_data = tensor_info.quant_param,
    };
    vx_tensor output_tensor = vxCreateTensor2(context_, &tensor_create_params,
                                              sizeof(tensor_create_params));
    if (output_tensor == nullptr) {
      throw std::runtime_error("Failed to create output vx tensor.");
    }

    vxSetParameterByIndex(nbg_node_, num_inputs + i,
                          reinterpret_cast<vx_reference>(output_tensor));
    output_tensors_.push_back(output_tensor);
  }

  status = vxVerifyGraph(graph_);
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to verify OpenVX graph.");
  }

  return static_cast<int>(status);
}

int OVXExecutor::query_nbg_io_infos() {
  uint32_t num_params;
  vxQueryKernel(nbg_kernel_, VX_KERNEL_PARAMETERS, &num_params,
                sizeof(num_params));

  for (uint32_t i = 0; i < num_params; i++) {
    vx_parameter param = vxGetKernelParameterByIndex(nbg_kernel_, i);

    vx_enum direction;
    vxQueryParameter(param, VX_PARAMETER_DIRECTION, &direction,
                     sizeof(direction));

    vx_meta_format meta;
    vxQueryParameter(param, VX_PARAMETER_META_FORMAT, &meta, sizeof(meta));

    OVXTensorInfo tensor_info;
    vxQueryMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS,
                               &tensor_info.rank, sizeof(tensor_info.rank));
    vxQueryMetaFormatAttribute(meta, VX_TENSOR_DIMS, tensor_info.shape.data(),
                               sizeof(tensor_info.shape));
    vxQueryMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE,
                               &tensor_info.data_type,
                               sizeof(tensor_info.data_type));
    vxQueryMetaFormatAttribute(meta, VX_TENSOR_QUANT_FORMAT,
                               &tensor_info.quant_type,
                               sizeof(tensor_info.quant_type));

    switch (tensor_info.quant_type) {
      case VX_QUANT_NONE:
        break;
      case VX_QUANT_AFFINE_SCALE:
        vxQueryMetaFormatAttribute(
            meta, VX_TENSOR_ZERO_POINT,
            &tensor_info.quant_param.affine.zeroPoint,
            sizeof(tensor_info.quant_param.affine.zeroPoint));
        vxQueryMetaFormatAttribute(
            meta, VX_TENSOR_SCALE, &tensor_info.quant_param.affine.scale,
            sizeof(tensor_info.quant_param.affine.scale));
        break;
      case VX_QUANT_DYNAMIC_FIXED_POINT:
        vxQueryMetaFormatAttribute(
            meta, VX_TENSOR_SCALE, &tensor_info.quant_param.dfp.fixed_point_pos,
            sizeof(tensor_info.quant_param.dfp.fixed_point_pos));
        break;
      default:
        vxReleaseParameter(&param);
        return VX_ERROR_NOT_SUPPORTED;
    }

    if (direction == VX_INPUT) {
      input_tensors_infos_.push_back(tensor_info);
    } else if (direction == VX_OUTPUT) {
      output_tensors_infos_.push_back(tensor_info);
    }

    vxReleaseParameter(&param);
  }

  return VX_SUCCESS;
}

int OVXExecutor::copy_to_input(size_t index, void* data, size_t rank,
                               const size_t* shape, const size_t* strides) {
  if (index >= input_tensors_infos_.size()) {
    throw std::out_of_range("Invalid input index.");
    return VX_FAILURE;
  }

  vx_tensor input_tensor = input_tensors_[index];
  auto tensor_info = input_tensors_infos_[index];

  if (rank != tensor_info.rank) {
    throw std::invalid_argument("Tensor rank mismatch.");
    return VX_FAILURE;
  }

  if (strides[0] != get_vx_dtype_bytes(tensor_info.data_type)) {
    throw std::invalid_argument("Tensor element size mismatch.");
    return VX_FAILURE;
  }

  for (size_t i = 0; i < rank; i++) {
    if (shape[i] != tensor_info.shape[i]) {
      throw std::invalid_argument("Tensor shape mismatch.");
      return VX_FAILURE;
    }
  }

  std::array<size_t, OVXTensorInfo::kMaxRank> view_start = {0};

  vx_status status =
      vxCopyTensorPatch(input_tensor, rank, view_start.data(), shape, strides,
                        data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to copy input data.");
    return status;
  }

  return VX_SUCCESS;
}

int OVXExecutor::copy_from_output(size_t index, void* data, size_t rank,
                                  const size_t* shape, const size_t* strides) {
  if (index >= output_tensors_infos_.size()) {
    throw std::runtime_error("Invalid output index.");
    return VX_FAILURE;
  }

  vx_tensor output_tensor = output_tensors_[index];

  std::array<size_t, OVXTensorInfo::kMaxRank> view_start = {0};
  vx_status status =
      vxCopyTensorPatch(output_tensor, rank, view_start.data(), shape, strides,
                        data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to copy output data.");
    return status;
  }

  return VX_SUCCESS;
}

int OVXExecutor::run() {
  vx_status status;

  status = vxProcessGraph(graph_);
  if (status != VX_SUCCESS) {
    throw std::runtime_error("Failed to run OpenVX graph.");
  }

  return static_cast<int>(status);
}

}  // namespace vsi::nbg_runner::vx
