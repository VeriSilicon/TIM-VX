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
#ifndef TIM_LAYOUT_INFER_TRANSPOSE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_TRANSPOSE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/transpose.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {
class TransposeLayoutInfer : public OpLayoutInfer {
 public:
  TransposeLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_input = op_->impl()->InputsTensor()[0];
    auto infer_input = context_->GetMapedTensor(src_input);
    auto input_pv = context_->GetPermuteVector(src_input);

    std::vector<uint32_t> perm(op_->impl()->node()->nn_param.permute.dim_num);
    memcpy(perm.data(), op_->impl()->node()->nn_param.permute.perm,
           op_->impl()->node()->nn_param.permute.dim_num * sizeof(uint32_t));
    IPermuteVectorPtr perm_pv = MakeShared(perm.size());
    for (uint32_t i = 0; i < perm.size(); i++) {
      perm_pv->At(i) = perm[i];
    }

    IPermuteVectorPtr final_pv = input_pv->Reverse()->Add(perm_pv);

    if (final_pv->IsAligned()) {
      //skip transpose op by insert a dummy reshape
      // context_->UpdateTensorMap(op_->impl()->OutputsTensor()[0], infer_input);
      auto reshape_op =
          context_->infer_graph_->CreateOperation<tim::vx::ops::Reshape>(
              op_->impl()->OutputsTensor()[0]->GetShape());
      reshape_op->BindInput(infer_input);
      auto reshape_out = CreateOutputsTensor(final_pv);
      reshape_op->BindOutput(reshape_out[0]);
    } else {
      auto transpose_op =
          context_->infer_graph_->CreateOperation<tim::vx::ops::Transpose>(
              final_pv->AsStdVec());
      transpose_op->BindInput(infer_input);
      //  The layout after final_pv permute is the default sequence
      auto infer_out = CreateOutputsTensor(MakeShared(perm.size()));
      transpose_op->BindOutput(infer_out[0]);
    }
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], MakeShared(perm.size()));
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim
#endif