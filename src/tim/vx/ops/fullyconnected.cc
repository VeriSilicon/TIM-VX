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
#include "tim/vx/ops/fullyconnected.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

FullyConnected::FullyConnected(Graph* graph, uint32_t axis)
    : FullyConnected(graph, axis, 0) {
}

FullyConnected::FullyConnected(Graph* graph, uint32_t axis, uint32_t weights)
    : BuiltinOp(graph, VSI_NN_OP_FCL2), axis_(axis), weights_(weights) {
  this->impl()->node()->nn_param.fcl.axis = axis;
  this->impl()->node()->nn_param.fcl.weights = weights;
}

std::shared_ptr<Operation> FullyConnected::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<FullyConnected>(this->axis_, this->weights_);
}

void FullyConnected::OnBindInputPostProc(const std::shared_ptr<Tensor>& tensor,
                           int32_t input_idx) {
  if (tensor->GetDataType() == vx::DataType::FLOAT16 &&
      tensor->IsConstTensor() && impl_->inputs_tensor_.size() == 3) {
    float* float32_bias = tensor->ConvertTensorToFloat32Data();

    TensorSpec fp32bias_spec(tim::vx::DataType::FLOAT32, tensor->GetShape(),
                             tim::vx::TensorAttribute::CONSTANT);

    auto out_tensor = impl_->graph_->CreateTensor(fp32bias_spec, float32_bias);
    vsi_nn_Free(float32_bias);

    impl_->inputs_tensor_[2] = out_tensor;
    impl_->node()->input.tensors[input_idx] = out_tensor->GetId();
    impl_->graph_->RenewTensorConsumersMap(tensor, out_tensor, this);
  }
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
