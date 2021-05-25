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
#ifndef TIM_LAYOUT_INFER_STRIDEDSLICE_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_STRIDEDSLICE_LAYOUT_INFERENCE_H_

#include "tim/vx/ops/stridedslice.h"

#include "src/tim/transform/ops/op_layout_inference.h"
#include "src/tim/transform/permute_vector.h"
#include "src/tim/vx/operation_private.h"

namespace tim {
namespace transform {
class StridedSliceLayoutInfer : public OpLayoutInfer {
 public:
  StridedSliceLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : OpLayoutInfer(op, context) {}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto src_input = op_->impl()->InputsTensor()[0];
    auto input_pv = context_->GetPermuteVector(src_input);

    int32_t begin_mask = op_->impl()->node()->nn_param.strided_slice.begin_mask;
    int32_t end_mask = op_->impl()->node()->nn_param.strided_slice.end_mask;
    int32_t shrink_axis_mask =
        op_->impl()->node()->nn_param.strided_slice.shrink_axis_mask;
    uint32_t begin_dims_num =
        op_->impl()->node()->nn_param.strided_slice.begin_dims_num;
    std::vector<int32_t> begin_dims(begin_dims_num);
    memcpy(begin_dims.data(),
           op_->impl()->node()->nn_param.strided_slice.begin_dims,
           begin_dims_num * sizeof(uint32_t));
    uint32_t end_dims_num =
        op_->impl()->node()->nn_param.strided_slice.end_dims_num;
    std::vector<int32_t> end_dims(end_dims_num);
    memcpy(end_dims.data(),
           op_->impl()->node()->nn_param.strided_slice.end_dims,
           end_dims_num * sizeof(uint32_t));
    uint32_t stride_dims_num =
        op_->impl()->node()->nn_param.strided_slice.stride_dims_num;
    std::vector<int32_t> stride_dims(stride_dims_num);
    memcpy(stride_dims.data(),
           op_->impl()->node()->nn_param.strided_slice.stride_dims,
           stride_dims_num * sizeof(uint32_t));

    begin_dims = MapMultipleAxis(input_pv->AsStdVec(), begin_dims);
    end_dims = MapMultipleAxis(input_pv->AsStdVec(), end_dims);
    stride_dims = MapMultipleAxis(input_pv->AsStdVec(), stride_dims);

    auto strided_slice =
        context_->infer_graph_->CreateOperation<vx::ops::StridedSlice>(
            begin_dims, end_dims, stride_dims, begin_mask, end_mask,
            shrink_axis_mask);
    auto infer_out = CreateOutputsTensor(input_pv);
    (*strided_slice).BindInput(context_->GetMapedTensor(src_input));
    (*strided_slice).BindOutput(infer_out[0]);

    context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], input_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace transform
}  // namespace tim
#endif