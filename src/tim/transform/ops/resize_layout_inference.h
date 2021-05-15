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
#ifndef TIM_LAYOUT_INFER_RESIZE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_RESIZE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/resize.h"

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"
namespace tim {
namespace transform {
class ResizeLayoutInfer : public OpLayoutInfer {
 public:
  ResizeLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    assert(op_->impl()->InputsTensor().size() == 1);
    vx::DataLayout layout = op_->impl()->layout_;
    auto required_pv = MakeShared(4);
    if (layout == vx::DataLayout::CWHN) {
      required_pv = std::make_shared<PermuteVector<4>>(kCWHN2WHCN);
    }
    auto i_src = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(i_src);
    auto final_pv = input_pv->Reverse()->Add(required_pv);

    if (!final_pv->IsAligned()) {
      auto perm_out = InsertPermute(i_src, final_pv);
      context_->UpdateTensorMap(i_src, perm_out);
      context_->SetPermuteVector(i_src, final_pv);
    }

    auto resize_type =
        static_cast<vx::ResizeType>(op_->impl()->node()->nn_param.resize.type);
    auto factor = op_->impl()->node()->nn_param.resize.factor;
    auto aglin_corners = op_->impl()->node()->nn_param.resize.align_corners;
    auto half_pixel_centers =
        op_->impl()->node()->nn_param.resize.half_pixel_centers;
    auto target_width = op_->impl()->node()->nn_param.resize.size[0];
    auto target_height = op_->impl()->node()->nn_param.resize.size[1];

    auto resize = context_->infer_graph_->CreateOperation<vx::ops::Resize>(
        resize_type, factor, aglin_corners, half_pixel_centers, target_height,
        target_width);

    auto out_infer = CreateOutputsTensor(required_pv);
    (*resize).BindInput(context_->GetMapedTensor(i_src));
    (*resize).BindOutput(out_infer[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace transform
}  // namespace tim

#endif