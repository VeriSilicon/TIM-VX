/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#ifdef TIM_VX_ENABLE_CUSTOM_OP
#include <map>
#include <assert.h>
#include "tim/vx/ops.h"
#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

#include "kernel/vsi_nn_kernel.h"

namespace tim {
namespace vx {
namespace ops {

static vsi_bool op_setup(vsi_nn_node_t* self, vsi_nn_tensor_t** inputs,
                         vsi_nn_tensor_t** outputs);

static vsi_bool op_compute(vsi_nn_node_t* self, vsi_nn_tensor_t** inputs,
                           vsi_nn_tensor_t** outputs);

static vx_status derive_kernel_init(vx_node node, const vx_reference* param,
                                    vx_uint32 param_size);

static std::map<void*, CustomOpBase*> node_base_map_;

CustomOpBase::CustomOpBase(Graph* graph, uint32_t input_num,
                           uint32_t output_num, int32_t kernel_id,
                           const char* kernel_name)
    : input_num_(input_num), output_num_(output_num) {
  init_kernel_ = reinterpret_cast<void*>(derive_kernel_init);
  vsi_nn_op_proc_t proc = {NULL,     op_compute, NULL,       NULL,
                           op_setup, NULL,       input_num_, output_num_};
  this->impl() = std::make_unique<CustomOpBaseImpl>(
      graph, kernel_id, reinterpret_cast<void*>(&proc), kernel_name);
  this->impl()->node()->nn_param.client_param = reinterpret_cast<void*>(this);
}

CustomOpBase::~CustomOpBase(){
  auto iter = node_base_map_.find(this->vx_node_);
  if (iter != node_base_map_.end()) {
    node_base_map_.erase(this->vx_node_);
  }
}
vsi_bool op_setup(vsi_nn_node_t* self, vsi_nn_tensor_t** inputs,
                  vsi_nn_tensor_t** outputs) {
  CustomOpBase* op_this =
      reinterpret_cast<CustomOpBase*>(self->nn_param.client_param);

  for (uint32_t i = 0; i < op_this->output_num_; i++) {
    std::vector<uint32_t> output_size;
    op_this->outputs_size_.push_back(output_size);
  }

  for (uint32_t i = 0; i < op_this->input_num_; i++) {
    std::vector<uint32_t> input_size;
    for (uint32_t j = 0; j < inputs[i]->attr.dim_num; j++) {
      input_size.push_back(inputs[i]->attr.size[j]);
    }
    op_this->inputs_size_.push_back(input_size);
  }

  op_this->SetupShapeInfor();

  for (uint32_t i = 0; i < op_this->outputs_size_.size(); i++) {
    outputs[i]->attr.dim_num = op_this->outputs_size_[i].size();
    for (uint32_t j = 0; j < op_this->outputs_size_[i].size(); j++) {
      outputs[i]->attr.size[j] = op_this->outputs_size_[i][j];
    }
  }
  return TRUE;
};

vsi_bool op_compute(vsi_nn_node_t* self, vsi_nn_tensor_t** inputs,
                    vsi_nn_tensor_t** outputs) {
  vsi_status status = VSI_FAILURE;
  auto kernel = vsi_nn_KernelCreate(VSI_NN_KERNEL_TYPE_CL);
  CustomOpBase* op_this =
      reinterpret_cast<CustomOpBase*>(self->nn_param.client_param);

  uint32_t param_num = op_this->param_list_.size();
  uint32_t input_start = op_this->input_num_ + op_this->output_num_;

  std::vector<tim::vx::DataType> input_types;
  for (uint32_t i = 0; i < op_this->input_num_; i++) {
    if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_FLOAT32) {
      input_types.push_back(tim::vx::DataType::FLOAT32);
    } else if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_UINT32) {
      input_types.push_back(tim::vx::DataType::UINT32);
    } else if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_INT32) {
      input_types.push_back(tim::vx::DataType::INT32);
    } else if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_BOOL8) {
      input_types.push_back(tim::vx::DataType::BOOL8);
    } else if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_UINT8) {
      input_types.push_back(tim::vx::DataType::UINT8);
    } else if (inputs[i]->attr.dtype.vx_type == VSI_NN_TYPE_INT8) {
      input_types.push_back(tim::vx::DataType::INT8);
    } else {
        std::cout << "Can not find att type in op compute" << std::endl;
        assert(false);
    }
  }

  std::string build_option;
  op_this->SetupParams(input_types, build_option);

  snprintf(kernel->info.name, VX_MAX_KERNEL_NAME, "%s", op_this->func_name_);
  kernel->unique_id =
      std::hash<std::string>()(std::string(op_this->func_name_));
  vx_param_description_t kernel_param_def[param_num + input_start];

  for (uint32_t i = 0; i < op_this->input_num_; i++) {
    kernel_param_def[i] = {VX_INPUT, VX_TYPE_TENSOR,
                           VX_PARAMETER_STATE_REQUIRED};
  }
  for (uint32_t i = 0; i < op_this->output_num_; i++) {
    kernel_param_def[op_this->input_num_ + i] = {VX_OUTPUT, VX_TYPE_TENSOR,
                                                 VX_PARAMETER_STATE_REQUIRED};
  }

  for (uint32_t i = 0; i < param_num; i++) {
    kernel_param_def[op_this->input_num_ + op_this->output_num_ + i] = {
        VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED};
  }

  kernel->info.parameters = kernel_param_def;
  kernel->info.enumeration = KERNEL_ID_PLACEHOLDER;
  kernel->info.numParams = param_num + input_start;
  kernel->info.initialize =
      reinterpret_cast<vx_kernel_initialize_f>(op_this->init_kernel_);

  vsi_nn_KernelAddSource(kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                           "executable_name");

  vsi_nn_KernelAddSource(kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2, "helper",
                           "fmt_code_name");

  const char* tmp[] = {"", op_this->kernel_resource_};
  const char** resource = tmp;

  vsi_nn_KernelAddBuildOption(kernel, build_option.c_str());

  auto node = vsi_nn_KernelCreateNodeExt(self->graph, kernel, resource);
  if (node) {

    std::vector<vsi_nn_kernel_node_param_t> node_params(param_num + input_start);
    vsi_nn_kernel_node_param_t* node_params_ptr = node_params.data();
    vsi_nn_kernel_node_pack_io(node_params_ptr, param_num + input_start, inputs,
                               op_this->input_num_, outputs,
                               op_this->output_num_);

    for (uint32_t i = 0; i < op_this->param_list_.size(); i++) {
      if (op_this->param_list_[i].type == tim::vx::DataType::FLOAT32) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, F32, &(op_this->param_list_[i].data.f));
      } else if (op_this->param_list_[i].type == tim::vx::DataType::UINT32) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, U32, &(op_this->param_list_[i].data.ui));
      } else if (op_this->param_list_[i].type == tim::vx::DataType::INT32) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, I32, &(op_this->param_list_[i].data.i));
      } else if (op_this->param_list_[i].type == tim::vx::DataType::BOOL8) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, BOOL8, &(op_this->param_list_[i].data.b));
      }else if (op_this->param_list_[i].type == tim::vx::DataType::UINT8) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, U8, &(op_this->param_list_[i].data.b));
      } else if (op_this->param_list_[i].type == tim::vx::DataType::INT8) {
        node_params_ptr[input_start++] = vsi_nn_kernelScalarCreate(
            self->graph, I8, &(op_this->param_list_[i].data.b));
      } else{
          std::cout << "Can not find scalar type in op compute" << std::endl;
          assert(false);
      }
    }

    input_start = op_this->input_num_ + op_this->output_num_;
    status = vsi_nn_KernelNodePassParam(node, node_params_ptr, param_num + input_start);
    for (uint32_t i = 0; i < param_num; i++) {
      vsi_nn_kernel_scalar_release(&node_params_ptr[input_start + i]);
    }

  }
  self->n = (vx_node)node;

  node_base_map_.insert(std::pair<void*, CustomOpBase*>(reinterpret_cast<void*>(self->n), op_this));
  op_this->vx_node_ = reinterpret_cast<void*>(self->n);
  return status;
}

vx_status derive_kernel_init(vx_node node, const vx_reference* param,
                             vx_uint32 param_size) {
  vsi_status status = VSI_FAILURE;
  if (param_size == 0 && param == nullptr) {
    return status;
  }

  gpu_param_t gpu_param = {3, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

  std::vector<size_t> global_size(3);
  std::vector<size_t> local_size(3);
  uint32_t dim = 0;

  auto iter = node_base_map_.find(reinterpret_cast<void*>(node));
  if (iter != node_base_map_.end()) {
    iter->second->SetupEnqueue(dim, global_size, local_size);
  } else {
    std::cout << "Something wrong in finding gpu param setup function"
              << std::endl;
    assert(false);
  }

  gpu_param.dim = dim;
  gpu_param.global_scale[0] = 1;
  gpu_param.global_scale[1] = 1;
  gpu_param.global_scale[2] = 1;

  gpu_param.global_size[0] = global_size[0];
  gpu_param.global_size[1] = global_size[1];
  gpu_param.global_size[2] = global_size[2];

  gpu_param.local_size[0] = local_size[0];
  gpu_param.local_size[1] = local_size[1];
  gpu_param.local_size[2] = local_size[2];
  status = vsi_nn_KernelGpuConfig(node, &gpu_param);

  return status;
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif