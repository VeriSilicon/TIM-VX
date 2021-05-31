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
#ifndef TIM_LAYOUT_INFER_GATHER_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_GATHER_LAYOUT_INFERENCE_H_

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/vx/operation_private.h"
#include "tim/vx/ops/gather.h"

namespace tim {
namespace transform {
class GatherLayoutInfer : public OpLayoutInfer {
 public:
  GatherLayoutInfer(
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    ReverseInputsPermuteVector();

    auto gather = context_->infer_graph_->CreateOperation<vx::ops::Gather>(
        op_->impl()->node()->nn_param.gather.axis);
    int32_t output_rank = -1;
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*gather).BindInput(context_->GetMapedTensor(i_src));
      output_rank += i_src->GetShape().size();
    }
    auto infer_out = CreateOutputsTensor(
        context_->GetPermuteVector(op_->impl()->InputsTensor()[0]));
    (*gather).BindOutput(infer_out[0]);
    auto output_pv = MakeShared(output_rank);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], output_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace transform
}  // namespace tim
#endif