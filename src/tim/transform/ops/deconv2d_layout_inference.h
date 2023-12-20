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
#ifndef TIM_LAYOUT_INFER_DECONV2D_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_DECONV2D_LAYOUT_INFERENCE_H_

#include "ops/op_layout_inference.h"
#include "permute_vector.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/deconv.h"

namespace tim {
namespace transform {
class DeConv2dLayoutInfer : public OpLayoutInfer {
 public:
  DeConv2dLayoutInfer(
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_deconv2d = std::static_pointer_cast<vx::ops::DeConv2d>(op_);
    vx::DataLayout layout = op_->impl()->layout_;
    auto kernel_layout = src_deconv2d->KernelDataLayout();
    std::shared_ptr<IPermuteVector> required_pv, weight_required_pv;
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
    switch (kernel_layout) {
      case vx::DataLayout::OcIcWH:  // Support TVM Kernel Layout
        weight_required_pv = std::make_shared<PermuteVector<4>>(kOcIcWH2WHIcOc);
        break;
      case vx::DataLayout::IcOcWH:
        weight_required_pv = std::make_shared<PermuteVector<4>>(kIcOcWH2WHIcOc);
        break;
      case vx::DataLayout::IcWHOc:  // Support nnapi & tflite Kernel Layout
        weight_required_pv = std::make_shared<PermuteVector<4>>(kIcWHOc2WHIcOc);
        break;
      default:
        weight_required_pv = std::make_shared<PermuteVector<4>>();
        break;
    }

    auto input_tensors = op_->impl()->InputsTensor();
    std::shared_ptr<vx::Tensor> infer_input, infer_weight, infer_bias;
    // For input
    auto input_pv = context_->GetPermuteVector(input_tensors[0]);
    auto final_pv = input_pv->Reverse()->Add(required_pv);
    if (!final_pv->IsAligned()) {
      infer_input =
          InsertPermute(context_->GetMapedTensor(input_tensors[0]), final_pv);
      context_->SetPermuteVector(input_tensors[0], required_pv);
    } else {
      infer_input = context_->GetMapedTensor(input_tensors[0]);
      context_->SetPermuteVector(input_tensors[0], input_pv);
    }
    context_->UpdateTensorMap(input_tensors[0], infer_input);

    // For weight
    if (input_tensors[1]->IsConstTensor()) {
      if (!weight_required_pv->IsAligned()) {
        infer_weight = PermuteConstTensor(input_tensors[1], weight_required_pv);
      } else {
        std::vector<uint8_t> dataRef(input_tensors[1]->GetSpec().GetByteSize());
        input_tensors[1]->CopyDataFromTensor(dataRef.data());
        infer_weight = context_->infer_graph_->CreateTensor(
            input_tensors[1]->GetSpec(), (const void*)dataRef.data());
      }
      context_->SetPermuteVector(input_tensors[1], weight_required_pv);
      context_->UpdateTensorMap(input_tensors[1], infer_weight);
    } else {
      auto weight_pv = context_->GetPermuteVector(input_tensors[1]);
      auto final_pv = weight_pv->Reverse()->Add(weight_required_pv);
      if (!final_pv->IsAligned()) {
        infer_weight =
            InsertPermute(context_->GetMapedTensor(input_tensors[1]), final_pv);
        context_->SetPermuteVector(input_tensors[1], weight_required_pv);
      } else {
        infer_weight = context_->GetMapedTensor(input_tensors[1]);
        context_->SetPermuteVector(input_tensors[1], weight_pv);
      }
      context_->UpdateTensorMap(input_tensors[1], infer_weight);
    }

    // For bias
    if (input_tensors.size() == 3) {
      if (input_tensors[2]->IsConstTensor()) {
        std::vector<uint8_t> dataRef(input_tensors[2]->GetSpec().GetByteSize());
        input_tensors[2]->CopyDataFromTensor(dataRef.data());
        infer_bias = context_->infer_graph_->CreateTensor(
            input_tensors[2]->GetSpec(), (const void*)dataRef.data());
      } else {
        infer_bias = context_->GetMapedTensor(input_tensors[2]);
      }
      auto bias_pv = MakeShared(1);
      context_->UpdateTensorMap(input_tensors[2], infer_bias);
      context_->SetPermuteVector(input_tensors[2], bias_pv);
    }

    auto deconv = op_->Clone(context_->infer_graph_);
    auto infer_out = CreateOutputsTensor(required_pv);
    for (const auto& i_src : input_tensors) {
      (*deconv).BindInput(context_->GetMapedTensor(i_src));
    }
    (*deconv).BindOutput(infer_out[0]);

    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif