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
#ifndef TIM_LAYOUT_INFER_ACTIVATION_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_ACTIVATION_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/reshape.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/transpose.h"

namespace tim {
namespace transform {
template <typename OpType>
class ActivationLayoutInfer : public OpLayoutInfer {
 public:
  ActivationLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    // Transmit input pv to out pv directly for activation ops
    assert(op_->impl()->InputsTensor().size() == 1);
    auto i_src = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(i_src);
    auto activation = op_->Clone(context_->infer_graph_);
    auto out_infer = CreateOutputsTensor(input_pv);
    (*activation)
        .BindInput(context_->GetMapedTensor(i_src))
        .BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

class PReluLayoutInfer : public OpLayoutInfer {
 public:
  PReluLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_input = op_->impl()->InputsTensor()[0];
    auto input_shape = src_input->GetShape();
    auto src_slope = op_->impl()->InputsTensor()[1];
    auto slope_shape = src_slope->GetShape();
    auto input_pv = context_->GetPermuteVector(src_input);
    std::vector<uint32_t> boardcast_shape;
    for (uint32_t i = 0; i < input_shape.size(); ++i) {
      if (i < slope_shape.size()) {
        boardcast_shape.push_back(slope_shape[i]);
      } else {
        boardcast_shape.push_back(1);
      }
    }

    if (src_slope->IsConstTensor()) {
      std::vector<uint8_t> dataRef(src_slope->GetSpec().GetByteSize());
      src_slope->CopyDataFromTensor(dataRef.data());
      auto infer_slope_spec = src_slope->GetSpec();
      infer_slope_spec.SetShape(boardcast_shape);
      auto infer_slope = context_->infer_graph_->CreateTensor(
          infer_slope_spec, (const void*)dataRef.data());

      if (!input_pv->IsAligned()) {
        //The dimension of slop is already the same as input, directly use input_pv to convert
        auto out_slope = PermuteConstTensor(infer_slope, input_pv);
        context_->UpdateTensorMap(src_slope, out_slope);
      } else {
        context_->UpdateTensorMap(src_slope, infer_slope);
      }
    } else {
      auto infer_slope_spec = src_slope->GetSpec().AsTransientSpec();
      auto reshape_out = context_->infer_graph_->CreateTensor(infer_slope_spec);
      boardcast_shape = MapMultipleAxis(input_pv->AsStdVec(), boardcast_shape);
      auto reshape = context_->infer_graph_->CreateOperation<vx::ops::Reshape>(boardcast_shape);
      (*reshape)
          .BindInput(context_->GetMapedTensor(src_slope))
          .BindOutput(reshape_out);
      context_->UpdateTensorMap(src_slope, reshape_out);
    }
    context_->SetPermuteVector(src_slope, input_pv);

    auto axis =
        MapAxis(input_pv->AsStdVec(), op_->impl()->node()->nn_param.prelu.axis);
    auto prelu = context_->infer_graph_->CreateOperation<vx::ops::Prelu>(axis);
    auto out_infer = CreateOutputsTensor(input_pv);

    (*prelu)
        .BindInput(context_->GetMapedTensor(src_input))
        .BindInput(context_->GetMapedTensor(src_slope));
    (*prelu).BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

using ReluLayoutInfer = ActivationLayoutInfer<vx::ops::Relu>;
using Relu1LayoutInfer = ActivationLayoutInfer<vx::ops::Relu1>;
using Relu6LayoutInfer = ActivationLayoutInfer<vx::ops::Relu6>;
using LeakyReluLayoutInfer = ActivationLayoutInfer<vx::ops::LeakyRelu>;
using EluLayoutInfer = ActivationLayoutInfer<vx::ops::Elu>;
using SigmoidLayoutInfer = ActivationLayoutInfer<vx::ops::Sigmoid>;
using MishLayoutInfer = ActivationLayoutInfer<vx::ops::Mish>;
using HardSigmoidLayoutInfer = ActivationLayoutInfer<vx::ops::HardSigmoid>;
using SoftReluLayoutInfer = ActivationLayoutInfer<vx::ops::SoftRelu>;
using HardSwishLayoutInfer = ActivationLayoutInfer<vx::ops::HardSwish>;
using TanhLayoutInfer = ActivationLayoutInfer<vx::ops::Tanh>;

}  // namespace transform
}  // namespace tim

#endif