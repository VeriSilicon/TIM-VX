/****************************************************************************
 *
 *    Copyright (c) 2023 Vivante Corporation
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
#ifndef TIM_BATCH_FUSE_TRANSPOSE_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_TRANSPOSE_BATCH_FUSE_H_

#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/slice.h"
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"

namespace tim {
namespace fuse {
class TransposeBatchFuse : public OpBatchFuse {
 public:
  TransposeBatchFuse(
      const std::shared_ptr<vx::Operation> op,
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
    auto gap = context_->GetForwardGap(input_tensor);

    //Shape of input and output is different, but their gap is the same
    context_->UpdateForwardGap(output_tensor, gap);
    next_tensors.push_back(output_tensor);

    //Transpose do not need backward
    return false;
  }

  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    //To hack a werror
    former_tensors.push_back(input_tensor);
    former_tensors.pop_back();
    return false;
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto input_spec = input_tensor->GetSpec();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_sepc = output_tensor->GetSpec();
    auto output_shape = output_tensor->GetShape();

    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    std::vector<uint32_t> perm(op_->impl()->node()->nn_param.permute.dim_num);
    memcpy(perm.data(), op_->impl()->node()->nn_param.permute.perm,
           op_->impl()->node()->nn_param.permute.dim_num * sizeof(uint32_t));
    auto transpose_op =
        context_->batch_fuse_graph_->CreateOperation<tim::vx::ops::Transpose>(
            perm);

    if (input_shape[batch_axis] != 1 &&
        input_batch_fuse_shape[batch_axis] == 1 &&
        output_sepc.attr_ == vx::TensorAttribute::OUTPUT) {
      auto concat_tensor = InsertSliceAndConcat(
          input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
      auto concat_shape = concat_tensor->GetShape();
      context_->UpdateTensorMap(input_tensor, concat_tensor);

      //Set tansposed shape according to permute parameters
      vx::ShapeType new_shape = {concat_shape[perm[0]], concat_shape[perm[1]],
                                 concat_shape[perm[2]], concat_shape[perm[3]]};
      output_sepc.SetShape(new_shape);
      auto out = context_->batch_fuse_graph_->CreateTensor(output_sepc);
      context_->UpdateProportion(input_tensor, 1);
      context_->UpdateTensorMap(output_tensor, out);

      (*transpose_op)
          .BindInput(context_->GetMapedTensor(input_tensor))
          .BindOutput(out);
      next_tensors.push_back(output_tensor);
    } else {
      //If it is not output, let it stay fused, no need to slice and concat
      vx::ShapeType new_shape = {
          //Set transposed shape according to permute parameters on fused shape
          input_batch_fuse_shape[perm[0]], input_batch_fuse_shape[perm[1]],
          input_batch_fuse_shape[perm[2]], input_batch_fuse_shape[perm[3]]};
      output_sepc.SetShape(new_shape);
      auto out = context_->batch_fuse_graph_->CreateTensor(output_sepc);
      context_->UpdateTensorMap(output_tensor, out);
      (*transpose_op)
          .BindInput(context_->GetMapedTensor(input_tensor))
          .BindOutput(out);
      next_tensors.push_back(output_tensor);
      auto valid_ratio = (float)(input_shape[w_axis] * input_shape[h_axis] *
                                 input_shape[batch_axis]) /
                         (float)(input_batch_fuse_shape[w_axis] *
                                 input_batch_fuse_shape[h_axis]);
      context_->UpdateProportion(input_tensor, valid_ratio);
    }
  }
};
}  // namespace fuse
}  // namespace tim

#endif