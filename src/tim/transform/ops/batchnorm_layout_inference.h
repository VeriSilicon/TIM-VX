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
#ifndef TIM_LAYOUT_INFER_BATCHNORM_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_BATCHNORM_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/batchnorm.h"

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "op_impl.h"
namespace tim {
namespace transform {
class BatchNormLayoutInfer : public OpLayoutInfer {
 public:
  BatchNormLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    vx::DataLayout layout = op_->impl()->layout_;
    auto required_pv = MakeShared(4);
    if (layout == vx::DataLayout::CWHN) {
      required_pv = std::make_shared<PermuteVector<4>>(kCWHN2WHCN);
    }
    auto input_tensors = op_->impl()->InputsTensor();
    assert(input_tensors.size() == 5);
    for (uint32_t idx = 0; idx < input_tensors.size(); idx++) {
        std::shared_ptr<vx::Tensor> perm_out;
        std::shared_ptr<IPermuteVector> input_pv;
        auto src_in = input_tensors[idx];
        if (src_in->IsConstTensor()) {
            std::vector<uint8_t> dataRef(src_in->GetSpec().GetByteSize());
            src_in->CopyDataFromTensor(dataRef.data());
            perm_out = context_->infer_graph_->CreateTensor(src_in->GetSpec(), (const void*)dataRef.data());
            input_pv = MakeShared(src_in->GetShape().size());
        } else {
          perm_out = context_->GetMapedTensor(src_in);
          input_pv = context_->GetPermuteVector(src_in);
          context_->SetPermuteVector(src_in, input_pv);
          if (idx == 0) {
            auto final_pv = input_pv->Reverse()->Add(required_pv);
            if (!final_pv->IsAligned()) {
              perm_out = InsertPermute(perm_out, required_pv);
              context_->SetPermuteVector(src_in, required_pv);
            }
          }
        }
        context_->UpdateTensorMap(src_in, perm_out);
    }

    auto batchnorm = op_->Clone(context_->infer_graph_);
    auto out_tensor_infer = CreateOutputsTensor(required_pv);
    (*batchnorm).BindInput(context_->GetMapedTensor(input_tensors[0]));
    (*batchnorm).BindInput(context_->GetMapedTensor(input_tensors[1]));
    (*batchnorm).BindInput(context_->GetMapedTensor(input_tensors[2]));
    (*batchnorm).BindInput(context_->GetMapedTensor(input_tensors[3]));
    (*batchnorm).BindInput(context_->GetMapedTensor(input_tensors[4]));

    (*batchnorm).BindOutput(out_tensor_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif