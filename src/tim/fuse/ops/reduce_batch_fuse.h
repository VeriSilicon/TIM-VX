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
#ifndef TIM_BATCH_FUSE_REDUCE_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_REDUCE_BATCH_FUSE_H_

#include "tim/vx/ops/reduce.h"
#include <set>
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"

namespace tim {
namespace fuse {
template <typename OpType>
class ReduceBatchFuse : public OpBatchFuse {
 public:
  ReduceBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();

    //Reduce op cannot be batch fused
    context_->UpdateGapInferShape(output_tensor, output_shape);
    context_->UpdateForwardGap(output_tensor, {0, 0});
    next_tensors.push_back(output_tensor);
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
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensor) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    std::shared_ptr<vx::Tensor> slice_and_concat_out;

    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);

    uint32_t batch = input_batch_fuse_shape[batch_axis];
    uint32_t batch_src = input_shape[batch_axis];
    if (batch == 1 && batch_src != 1) {
      //Reduce op need insert slice and concat
      slice_and_concat_out =
          InsertSliceAndConcat(input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
    } else {
      slice_and_concat_out = input_batch_fuse_tensor;
    }
    auto slice_and_concat_shape = slice_and_concat_out->GetShape();
    context_->UpdateProportion(input_tensor, 1);
    context_->UpdateTensorMap(input_tensor, slice_and_concat_out);

    //Get reduce axis
    std::vector<int32_t> new_axis;
    for (uint32_t i = 0; i < op_->impl()->node()->nn_param.reduce.axis_num;
         ++i) {
      int32_t axis = op_->impl()->node()->nn_param.reduce.axis[i];
      if (axis < 0) axis += input_shape.size(); //Handle negative axis
      new_axis.push_back(axis);
    }

    auto reduce = context_->batch_fuse_graph_->CreateOperation<OpType>(
        new_axis, op_->impl()->node()->nn_param.reduce.keep_dim);

    auto reduce_out_tensor = CreateOutputsTensor()[0];
    (*reduce).BindInput(slice_and_concat_out).BindOutput(reduce_out_tensor);
    next_tensor.push_back(output_tensor);
  }
};

using ReduceMinBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceMin>;
using ReduceMaxBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceMax>;
using ReduceAnyBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceAny>;
using ReduceProdBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceProd>;
using ReduceMeanBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceMean>;
using ReduceSumBatchFuse = ReduceBatchFuse<tim::vx::ops::ReduceSum>;
}  // namespace fuse
}  // namespace tim

#endif