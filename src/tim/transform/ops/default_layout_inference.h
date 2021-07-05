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

#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/batchnorm.h"
#include "tim/vx/ops/clip.h"


#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"

namespace tim {
namespace transform {

class DefaultLayoutInfer : public OpLayoutInfer {
 public:
  DefaultLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  // reverse any applied permute on it's input tensor
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    ReverseInputsPermuteVector();

    auto cloned_op = op_->Clone(context_->infer_graph_);

    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*cloned_op).BindInput(context_->GetMapedTensor(i_src));
    }
    auto required_pv =
        MakeShared(op_->impl()->OutputsTensor()[0]->GetShape().size());
    auto out_infer = CreateOutputsTensor(required_pv);

    // TODO: bind all output
    (*cloned_op).BindOutputs(out_infer);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif