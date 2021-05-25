/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "tim/vx/operation.h"

#include <vector>

#include "graph_private.h"
#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
OperationImpl::OperationImpl(Graph* graph, uint32_t operation_id,
                              int input_cnt, int output_cnt, DataLayout layout)
    : graph_(reinterpret_cast<GraphImpl*>(graph)),
      operation_id_(operation_id),
      input_cnt_(input_cnt),
      output_cnt_(output_cnt),
      layout_(layout),
      node_(vsi_nn_AddNode(graph_->graph(), operation_id_, input_cnt_,
                           output_cnt_, NULL)) {
  SetRoundingPolicy();
  node_->uid = graph_->graph()->cur_nid;
}

OperationImpl& OperationImpl::BindInput(const std::shared_ptr<Tensor>& tensor) {
  inputs_tensor_.push_back(tensor);
  uint32_t tensor_id = tensor->GetId();
  node_->input.tensors[input_tensor_index++] = tensor_id;
  if (tensor->GetSpec().attr_ & TensorAttribute::INPUT) {
    graph_->AddInput(tensor_id);
    graph_->AddInput(tensor);
  }
  return *this;
}

OperationImpl& OperationImpl::BindOutput(
    const std::shared_ptr<Tensor>& tensor) {
  outputs_tensor_.push_back(tensor);
  uint32_t tensor_id = tensor->GetId();
  node_->output.tensors[output_tensor_index++] = tensor_id;
  if (tensor->GetSpec().attr_ == TensorAttribute::OUTPUT) {
    graph_->AddOutput(tensor_id);
    graph_->AddOutput(tensor);
  }
  return *this;
}

OperationImpl& OperationImpl::SetRoundingPolicy(
    OverflowPolicy overflow_policy, RoundingPolicy rounding_policy,
    RoundType down_scale_size_rounding, uint32_t accumulator_bits) {
  node_->vx_param.overflow_policy = TranslateOverflowPolicy(overflow_policy);
  node_->vx_param.rounding_policy = TranslateRoundingPolicy(rounding_policy);
  node_->vx_param.down_scale_size_rounding =
      TranslateDownScaleSizeRounding(down_scale_size_rounding);
  node_->vx_param.accumulator_bits = accumulator_bits;

  return *this;
}

// Operation implementation
Operation::Operation(Graph* graph, uint32_t operation_id,
                     int input_cnt, int output_cnt,  DataLayout layout) {
  impl_ = std::make_unique<OperationImpl>(graph, operation_id,
                                          input_cnt, output_cnt, layout);
}

Operation::~Operation() {}

std::unique_ptr<OperationImpl>& Operation::impl() { return impl_; }

Operation& Operation::BindInput(const std::shared_ptr<Tensor>& tensor) {
  impl_->BindInput(tensor);
  impl_->graph_->UpdateTensorConsumersMap(tensor, this);
  return *this;
}

Operation& Operation::BindOutput(const std::shared_ptr<Tensor>& tensor) {
  impl_->BindOutput(tensor);
  return *this;
}

Operation& Operation::SetRoundingPolicy(
    OverflowPolicy overflow_policy, RoundingPolicy rounding_policy,
    RoundType down_scale_size_rounding, uint32_t accumulator_bits) {
  impl_->SetRoundingPolicy(overflow_policy, rounding_policy,
                           down_scale_size_rounding, accumulator_bits);
  return *this;
}

Operation& Operation::BindInputs(
    const std::vector<std::shared_ptr<Tensor>>& tensors) {
  for (auto& t : tensors) {
    BindInput(t);
  }
  return *this;
}

Operation& Operation::BindOutputs(
    const std::vector<std::shared_ptr<Tensor>>& tensors) {
  for (auto& t : tensors) {
    BindOutput(t);
  }
  return *this;
}

}  // namespace vx
}  // namespace tim