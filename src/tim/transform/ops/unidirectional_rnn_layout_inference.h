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
#ifndef TIM_LAYOUT_INFER_UNIDIRECTIONAL_RNN_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_UNIDIRECTIONAL_RNN_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/batchnorm.h"
#include "tim/vx/ops/clip.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {

class UnidirectionalRnnLayoutInfer : public OpLayoutInfer {
 public:
  UnidirectionalRnnLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  // reverse any applied permute on it's input tensor
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    ReverseInputsPermuteVector();

    auto cloned_op = op_->Clone(context_->infer_graph_);

    for (const auto& i_src : op_->impl()->InputsTensor()) {
      std::shared_ptr<vx::Tensor> infer_tensor;
      std::shared_ptr<IPermuteVector> required_pv;
      if ((i_src->IsConstTensor() &&
              !(i_src->GetSpec().attr_ & vx::TensorAttribute::INPUT))) {
        std::vector<uint8_t> dataRef(i_src->GetSpec().GetByteSize());
        i_src->CopyDataFromTensor(dataRef.data());
        infer_tensor = context_->infer_graph_->CreateTensor(
            i_src->GetSpec(), (const void*)dataRef.data());
        context_->UpdateTensorMap(i_src, infer_tensor);
      }
      if (i_src->GetId() == (uint32_t)-1) {
        infer_tensor = context_->infer_graph_->CreateTensorPlaceHolder();
        context_->UpdateTensorMap(i_src, infer_tensor);
      }
      required_pv = MakeShared(i_src->GetShape().size());
      context_->SetPermuteVector(i_src, required_pv);
    }


    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*cloned_op).BindInput(context_->GetMapedTensor(i_src));
    }

    
    std::vector<std::shared_ptr<IPermuteVector>> required_pv_lst;
    for (auto out_tensor : op_->impl()->OutputsTensor()) {
      std::shared_ptr<vx::Tensor> infer_tensor;
       if (out_tensor->GetId() == (uint32_t)-1) {
        out_tensor = context_->infer_graph_->CreateTensorPlaceHolder();
      }
      required_pv_lst.push_back(MakeShared(out_tensor->GetShape().size()));
    }
    auto out_infer = CreateOutputsTensor(required_pv_lst);

    (*cloned_op).BindOutputs(out_infer);
    uint32_t i = 0;
    for (auto out_tensor : op_->impl()->OutputsTensor()) {
      context_->SetPermuteVector(out_tensor, required_pv_lst[i++]);
      next_tensors.push_back(out_tensor);
    }
  }
};

}  // namespace transform
}  // namespace tim

#endif