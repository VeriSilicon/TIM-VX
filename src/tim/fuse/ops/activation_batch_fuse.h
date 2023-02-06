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
#ifndef TIM_BATCH_FUSE_ACTIVATION_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_ACTIVATION_BATCH_FUSE_H_

#include "tim/vx/ops/activations.h"

#include "op_batch_fuse.h"
// #include "permute_vector.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {

template <typename OpType>
class ActivationBatchFuse : public OpBatchFuse {
 public:
  ActivationBatchFuse(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}
  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto i_src = op_->impl()->InputsTensor()[0];
    auto i_src_shape = i_src->GetShape();
    // auto i_src_batch_fuse_spec = context_->GetMapedTensor(i_src)->GetSpec();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto pad = context_->GetForwardPad(i_src);
    auto i_infer_shape = context_->GetPadInferShape(i_src);
    // same as input
    context_->UpdateInitPad(o_src, {0, 0, 0, 0});
    context_->UpdateForwardPad(o_src, pad);
    context_->UpdatePadInferShape(o_src, i_infer_shape);

    auto gap = context_->GetForwardGap(i_src);
    context_->UpdateForwardGap(o_src, gap);
    next_tensors.push_back(o_src);
    return false;
  }

  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto i_src = op_->impl()->InputsTensor()[0];
    auto i_src_shape = i_src->GetShape();
    // auto i_src_batch_fuse_spec = context_->GetMapedTensor(i_src)->GetSpec();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto pad = context_->GetForwardPad(i_src);
    auto i_infer_shape = context_->GetPadInferShape(i_src);
    auto o_infer_shape = context_->GetPadInferShape(o_src);
    auto gap = context_->GetForwardGap(o_src);
    context_->UpdateForwardGap(i_src, gap);
    
    //pass new infered shape to input of activation
    context_->UpdatePadInferShape(i_src, o_infer_shape);
    
    former_tensors.push_back(i_src);

    //when activatetion is in backward, always need backward, but this flag is useless
    return true; 
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    assert(op_->impl()->InputsTensor().size() == 1);
    auto i_src = op_->impl()->InputsTensor()[0];
    auto i_src_shape = i_src->GetShape();
    auto i_src_batch_fuse_spec = context_->GetMapedTensor(i_src)->GetSpec();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto activation = op_->Clone(context_->batch_fuse_graph_);
    auto i_src_batch_fuse_shape = context_->GetMapedTensor(i_src)->GetShape();
    auto o_src_spec = o_src->GetSpec();
    auto out_batch_spec = o_src_spec.SetShape(i_src_batch_fuse_shape);
    auto out_batch_fuse =
        context_->batch_fuse_graph_->CreateTensor(out_batch_spec);
    // auto out_batch_fuse = CreateOutputsTensor();
    auto out_batch_fuse_shape = out_batch_fuse->GetShape();
    context_->UpdateTensorMap(o_src, out_batch_fuse);
    context_->UpdateTensorBatchFuseMap(out_batch_fuse, o_src);
    (*activation)
        .BindInput(context_->GetMapedTensor(i_src))
        .BindOutput(out_batch_fuse);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
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