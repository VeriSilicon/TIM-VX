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
#include "tim/vx/operation.h"
#include <vector>
#include "op_impl.h"

#include "graph_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
// Operation implementation
Operation::Operation() {}

Operation::~Operation() {}

std::unique_ptr<OpImpl>& Operation::impl() { return impl_; }
const std::unique_ptr<OpImpl>& Operation::impl() const { return impl_; }

Operation& Operation::BindInput(const std::shared_ptr<Tensor>& tensor) {
  impl_->BindInput(tensor);
  impl_->graph_->UpdateTensorConsumersMap(tensor, this);
  OnBindInputPostProc(tensor, impl_->input_tensor_index - 1);
  return *this;
}

Operation& Operation::BindOutput(const std::shared_ptr<Tensor>& tensor) {
  impl_->BindOutput(tensor);
  impl_->graph_->UpdateTensorProducerMap(tensor, this);
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

bool Operation::IsAllInputsConst() const{
  for (auto tensor : impl_->inputs_tensor_) {
    if (!tensor->IsConstTensor()) return false;
  }
  return true;
}

const std::vector<std::shared_ptr<Tensor>> Operation::ConstantInputsTensor() const{
  if (this->IsAllInputsConst()) {
    return impl_->inputs_tensor_;
  } else {
    return {};
  }
}
void Operation::OnBindInputPostProc(const std::shared_ptr<Tensor>& tensor, int32_t input_idx){
  (void) tensor;
  (void) input_idx;
}

}  // namespace vx
}  // namespace tim