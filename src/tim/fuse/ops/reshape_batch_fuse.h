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
#ifndef TIM_BATCH_FUSE_RESHAPE_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_RESHAPE_BATCH_FUSE_H_

#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/slice.h"
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {
class ReshapeBatchFuse : public OpBatchFuse {
 public:
  ReshapeBatchFuse(const std::shared_ptr<vx::Operation> op,
                   std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();

    auto input_gap_infer_shape = context_->GetGapInferShape(input_tensor);
    context_->UpdateGapInferShape(output_tensor, output_shape);

    //Set output's gap with input's gap
    auto gap_input = context_->GetForwardGap(input_tensor);
    context_->UpdateForwardGap(output_tensor, gap_input);
    next_tensors.push_back(output_tensor);
    return false;
  }

  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    //Hack a werror
    former_tensors.push_back(input_tensor);
    former_tensors.pop_back();
    return false;
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensor) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto input_spec = input_tensor->GetSpec();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_spec = output_tensor->GetSpec();
    auto reshape_op = op_->Clone(context_->batch_fuse_graph_);

    // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    // auto batch_src_axis = context_->GetBatchAxis();  // 3
    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    // auto c_axis = context_->GetPermChannelAxis(input_tensor);

    // auto w_axis = fuse_axes[0];
    // auto h_axis = fuse_axes[1];

    if (input_shape[batch_axis] != 1 && input_batch_fuse_shape[batch_axis] == 1) {
      if (output_spec.attr_ == vx::TensorAttribute::OUTPUT) {
        auto concat_tensor = InsertSliceAndConcat(input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
        context_->UpdateTensorMap(input_tensor, concat_tensor);
      }
    }

    //No batch fuse, so valid propartion is 1
    context_->UpdateProportion(input_tensor, 1);
    auto out = context_->batch_fuse_graph_->CreateTensor(output_spec);
    context_->UpdateTensorMap(output_tensor, out);
    (*reshape_op)
        .BindInput(context_->GetMapedTensor(input_tensor))
        .BindOutput(out);
    next_tensor.push_back(output_tensor);
  }
};
}  // namespace fuse
}  // namespace tim
#endif