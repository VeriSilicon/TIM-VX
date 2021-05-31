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
#ifndef TIM_LAYOUT_INFER_SLICE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_SLICE_LAYOUT_INFERENCE_H_

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/vx/operation_private.h"
#include "tim/vx/ops/slice.h"

namespace tim {
namespace transform {
class SliceLayoutInfer : public OpLayoutInfer {
 public:
  SliceLayoutInfer(
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_input = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(src_input);

    uint32_t dims = op_->impl()->node()->nn_param.slice.dims;
    const uint32_t* start_ptr = op_->impl()->node()->nn_param.slice.start;
    const uint32_t* length_ptr = op_->impl()->node()->nn_param.slice.length;
    std::vector<int32_t> start(dims);
    std::vector<int32_t> length(dims);
    memcpy(start.data(), start_ptr, dims * sizeof(uint32_t));
    memcpy(length.data(), length_ptr, dims * sizeof(uint32_t));
    start = MapMultipleAxis(input_pv->AsStdVec(), start);
    length = MapMultipleAxis(input_pv->AsStdVec(), length);

    auto slice = context_->infer_graph_->CreateOperation<vx::ops::Slice>(
        dims, start, length);
    auto infer_out = CreateOutputsTensor(input_pv);
    (*slice).BindInput(context_->GetMapedTensor(src_input));
    (*slice).BindOutput(infer_out[0]);
    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace transform
}  // namespace tim
#endif