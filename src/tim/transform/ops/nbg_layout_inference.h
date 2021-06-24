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
#ifndef TIM_LAYOUT_INFER_DEFAULT_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_DEFAULT_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/nbg.h"

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/vx/operation_private.h"

namespace tim {
namespace transform {
class NbgLayoutInfer : public OpLayoutInfer {
 public:
  NbgLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  // reverse any applied permute on it's input tensor
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    ReverseInputsPermuteVector();

    auto url = op_->impl()->node()->nn_param.nbg.url;
    uint32_t input_count = op_->impl()->input_cnt_;
    uint32_t output_count = op_->impl()->output_cnt_;
    auto nbg = context_->infer_graph_->CreateOperation<vx::ops::NBG>(
        url, input_count, output_count);

    for (auto i_src : op_->impl()->InputsTensor()) {
      (*nbg).BindInput(context_->GetMapedTensor(i_src));
      auto input_pv = MakeShared(i_src->GetShape().size());
      context_->SetPermuteVector(i_src, input_pv);
    }
    auto infer_out = CreateOutputsTensor(MakeShared(1));
    (*nbg).BindOutputs(infer_out);
    for (const auto& out : op_->impl()->OutputsTensor()) {
      context_->SetPermuteVector(out, MakeShared(out->GetShape().size()));
      next_tensors.push_back(out);
    }
  }
};

}  // namespace transform
}  // namespace tim

#endif