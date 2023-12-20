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
#ifndef TIM_LAYOUT_INFER_ROI_POOL_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_ROI_POOL_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/roi_pool.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace transform {

class RoiPoolLayoutInfer : public OpLayoutInfer {
 public:
  RoiPoolLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    vx::DataLayout layout = op_->impl()->layout_;
    auto input_tensors = op_->impl()->InputsTensor();
    std::shared_ptr<IPermuteVector> required_pv;
    switch (layout)
    { // kernel layout must be IWHO in tflite & nnapi
      case vx::DataLayout::CWHN:
        required_pv = std::make_shared<PermuteVector<4>>(kCWHN2WHCN);
        break;
      case vx::DataLayout::WHCN:
        required_pv = MakeShared(4);
        break;
      default:
        VSILOGE("The layout of input is not support.");
        required_pv = MakeShared(4);
        break;
    }
    auto input_pv = context_->GetPermuteVector(input_tensors[0]);
    auto final_pv = input_pv->Reverse()->Add(required_pv);
    std::shared_ptr<vx::Tensor> infer_input;
    if (!final_pv->IsAligned()) {
      infer_input = InsertPermute(context_->GetMapedTensor(input_tensors[0]), final_pv);
      context_->SetPermuteVector(input_tensors[0], required_pv);
    } else {
      infer_input = context_->GetMapedTensor(input_tensors[0]);
      context_->SetPermuteVector(input_tensors[0], input_pv);
    }
    context_->UpdateTensorMap(input_tensors[0], infer_input);

    for (const auto& t_src : op_->impl()->InputsTensor()) {
      if(t_src->IsConstTensor()) {
        std::vector<uint8_t> dataRef(t_src->GetSpec().GetByteSize());
        t_src->CopyDataFromTensor(dataRef.data());
        auto t_infer = context_->infer_graph_->CreateTensor(
                t_src->GetSpec(), (const void*)dataRef.data());
        context_->SetPermuteVector(t_src, MakeShared(t_src->GetShape().size()));
        context_->UpdateTensorMap(t_src, t_infer);
      }
    }

    auto roi_pool = op_->Clone(context_->infer_graph_);
    auto outs_infer = CreateOutputsTensor(required_pv);
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*roi_pool).BindInput(context_->GetMapedTensor(i_src));
    }
    (*roi_pool).BindOutput(outs_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif