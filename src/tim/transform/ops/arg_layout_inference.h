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
#ifndef TIM_LAYOUT_INFER_ARG_OPS_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_ARG_OPS_LAYOUT_INFERENCE_H_

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/vx/operation_private.h"
#include "tim/vx/ops/arg.h"
namespace tim {
namespace transform {
class ArgLayoutInfer : public OpLayoutInfer {
 public:
  ArgLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    ReverseInputsPermuteVector();
    assert(1 == op_->impl()->InputsTensor().size());
    auto src_input = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(src_input);

    auto arg = op_->Clone(context_->infer_graph_);
    auto infer_out = CreateOutputsTensor(input_pv);
    (*arg).BindInput(context_->GetMapedTensor(src_input));
    (*arg).BindOutput(infer_out[0]);

    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim
#endif