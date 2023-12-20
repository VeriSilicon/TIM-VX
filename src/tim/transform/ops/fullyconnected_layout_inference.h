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
#ifndef TIM_LAYOUT_INFER_FULLYCONNECTED_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_FULLYCONNECTED_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/fullyconnected.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {
class FullyConnectedLayoutInfer : public OpLayoutInfer {
 public:
  FullyConnectedLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    /* vx_delegate has reshaped the input to two-dimensional when mapping,
       The axis of 2-dimensional fc can only be 0. */
    auto input_tensors = op_->impl()->InputsTensor();
    if(!context_->GetPermuteVector(input_tensors[0])->IsAligned()){
      ReverseInputsPermuteVector();
    }
    for (const auto& in : input_tensors) {
      if (in->IsConstTensor()) {
        std::vector<uint8_t> dataRef(in->GetSpec().GetByteSize());
        in->CopyDataFromTensor(dataRef.data());
        auto infer_tensor = context_->infer_graph_->CreateTensor(in->GetSpec(),
                                                            (const void*)dataRef.data());
        auto trans_pv = MakeShared(in->GetShape().size());

        context_->UpdateTensorMap(in, infer_tensor);
        context_->SetPermuteVector(in, trans_pv);
      }
    }

    auto fcl = op_->Clone(context_->infer_graph_);
    auto required_pv =
        MakeShared(op_->impl()->OutputsTensor()[0]->GetShape().size());
    auto out_infer = CreateOutputsTensor(required_pv);
    for (auto in : op_->impl()->InputsTensor()) {
      (*fcl).BindInput(context_->GetMapedTensor(in));
    }
    (*fcl).BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim
#endif