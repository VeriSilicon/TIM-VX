/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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
#ifndef TIM_BATCH_FUSE_ACTIVATION_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_ACTIVATION_BATCH_FUSE_H_

#include "tim/vx/ops/activations.h"
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {

template <typename OpType>
class ActivationBatchFuse : public OpBatchFuse {
 public:
  ActivationBatchFuse(){};
  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];

    //the infered shape with gap inside
    auto input_infer_shape = context_->GetGapInferShape(input_tensor);

    //Activation will not be affected by batch fuse and gap inside,
    //so the gap and GapInferShape will be passed directly from input tensor to output tensor.
    context_->UpdateGapInferShape(output_tensor, input_infer_shape);
    auto gap = context_->GetForwardGap(input_tensor);

    context_->UpdateForwardGap(output_tensor, gap);
    next_tensors.push_back(output_tensor);

    //Activation always do not need backward
    return false;
  }

  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];

    //the infered shape with gap inside
    auto input_infer_shape = context_->GetGapInferShape(input_tensor);
    auto output_infer_shape = context_->GetGapInferShape(output_tensor);
    auto gap = context_->GetForwardGap(output_tensor);

    //pass new gap and infered shape to the input of activation
    context_->UpdateForwardGap(input_tensor, gap);
    context_->UpdateGapInferShape(input_tensor, output_infer_shape);

    former_tensors.push_back(input_tensor);

    //when activatetion is in backward, always need backward
    //if it is triggered by its next op which need backward
    //, but this flag is useless
    return true;
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    assert(op_->impl()->InputsTensor().size() == 1);
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_spec = output_tensor->GetSpec();

    auto fuse_src_axes = context_->GetFuseAxes();  // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();

    //fused output shape is the same as input shape
    auto output_batch_fuse_spec = output_spec.SetShape(input_batch_fuse_shape);
    auto output_batch_fuse_tensor =
        context_->GetBatchFuseGraph()->CreateTensor(output_batch_fuse_spec);

    //compute the proporation of valid data
    auto valid_prop = (float)(input_shape[w_axis] * input_shape[h_axis] *
                              input_shape[batch_axis]) /
                      (float)(input_batch_fuse_shape[w_axis] *
                              input_batch_fuse_shape[h_axis]);
    context_->UpdateProportion(input_tensor, valid_prop);
    context_->UpdateTensorMap(output_tensor, output_batch_fuse_tensor);

    //clone op to batch_fuse_graph_
    auto activation = op_->Clone(context_->GetBatchFuseGraph());
    (*activation)
        .BindInput(context_->GetMapedTensor(input_tensor))
        .BindOutput(output_batch_fuse_tensor);
    next_tensors.push_back(output_tensor);
  }
};

using ReluBatchFuse = ActivationBatchFuse<vx::ops::Relu>;
using Relu1BatchFuse = ActivationBatchFuse<vx::ops::Relu1>;
using Relu6BatchFuse = ActivationBatchFuse<vx::ops::Relu6>;
using LeakyReluBatchFuse = ActivationBatchFuse<vx::ops::LeakyRelu>;
using EluBatchFuse = ActivationBatchFuse<vx::ops::Elu>;
using SigmoidBatchFuse = ActivationBatchFuse<vx::ops::Sigmoid>;
using MishBatchFuse = ActivationBatchFuse<vx::ops::Mish>;
using HardSigmoidBatchFuse = ActivationBatchFuse<vx::ops::HardSigmoid>;
using SoftReluBatchFuse = ActivationBatchFuse<vx::ops::SoftRelu>;
using HardSwishBatchFuse = ActivationBatchFuse<vx::ops::HardSwish>;
using TanhBatchFuse = ActivationBatchFuse<vx::ops::Tanh>;

}  // namespace fuse
}  // namespace tim

#endif