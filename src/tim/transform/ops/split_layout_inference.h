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
#ifndef TIM_LAYOUT_INFER_SPLIT_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_SPLIT_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/split.h"

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"

namespace tim {
namespace transform {
class SplitLayoutInfer : public OpLayoutInfer {
 public:
  SplitLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    uint32_t slices_num = op_->impl()->node()->nn_param.split.slices_num;
    std::vector<uint32_t> slices(slices_num);
    memcpy(slices.data(), op_->impl()->node()->nn_param.split.slices,
           slices_num * sizeof(uint32_t));
    auto input_pv = context_->GetPermuteVector(input_tensor);
    uint32_t axis =
        MapAxis(input_pv->AsStdVec(), op_->impl()->node()->nn_param.split.axis);
    auto split =
        context_->infer_graph_->CreateOperation<vx::ops::Split>(axis, slices);
    auto infer_out = CreateOutputsTensor(input_pv);
    (*split).BindInput(context_->GetMapedTensor(input_tensor));
    (*split).BindOutputs(infer_out);
    for (const auto& out : op_->impl()->OutputsTensor()) {
        context_->SetPermuteVector(out, input_pv);
        next_tensors.push_back(out);
    }
  }
};
}  // namespace transform
}  // namespace tim
#endif