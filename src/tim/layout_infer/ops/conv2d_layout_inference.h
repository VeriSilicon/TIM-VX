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
#ifndef TIM_LAYOUT_INFER_CONV2D_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_CONV2D_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/conv2d.h"

#include "src/tim/vx/operation_private.h"
#include "src/tim/layout_infer/permute_vector.h"
#include "src/tim/layout_infer/ops/op_layout_inference.h"

namespace tim {
namespace transform {

class Conv2dLayoutInfer : public OpLayoutInfer {
 public:
  Conv2dLayoutInfer(
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

    // for input and weight
    for (uint32_t i = 0; i < 2; i++) {
      auto pv = context_->GetPermuteVector(input_tensors[i]);
      auto final_pv = pv->Reverse()->Add(required_pv);
      if (!final_pv->IsAligned()) {
        auto perm_out =
            InsertPermute(context_->GetMapedTensor(input_tensors[i]), final_pv);
        context_->UpdateTensorMap(input_tensors[i], perm_out);
        context_->SetPermuteVector(input_tensors[i], required_pv);
      }
    }

    auto pad_type = TranslatePadType(op_->impl()->node()->nn_param.conv2d.pad_type);
    std::array<uint32_t, 2> ksize = {
      op_->impl()->node()->nn_param.conv2d.ksize[0],
          op_->impl()->node()->nn_param.conv2d.ksize[1]
    };
    std::array<uint32_t, 2> stride = {
      op_->impl()->node()->nn_param.conv2d.stride[0],
      op_->impl()->node()->nn_param.conv2d.stride[1]
    };
    std::array<uint32_t, 2> dilation = {
      op_->impl()->node()->nn_param.conv2d.dilation[0],
      op_->impl()->node()->nn_param.conv2d.dilation[1]
    };
    std::array<uint32_t, 4> pad = {
      op_->impl()->node()->nn_param.conv2d.pad[0],
      op_->impl()->node()->nn_param.conv2d.pad[1],
      op_->impl()->node()->nn_param.conv2d.pad[2],
      op_->impl()->node()->nn_param.conv2d.pad[3]
    };
    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;
    auto conv2d = context_->infer_graph_->CreateOperation<vx::ops::Conv2d>(
        out_channels, pad_type, ksize, stride, dilation, pad, multiplier,
        vx::DataLayout::WHCN);
    auto otensor_infer = CreateOutputsTensor(required_pv);
    (*conv2d).BindInputs({context_->GetMapedTensor(input_tensors[0]),
                          context_->GetMapedTensor(input_tensors[1]),
                          context_->GetMapedTensor(input_tensors[2])});
    (*conv2d).BindOutput(otensor_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif