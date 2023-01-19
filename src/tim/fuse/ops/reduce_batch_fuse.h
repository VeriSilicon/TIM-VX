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
// #include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace fuse {
template <typename OpType>
class ReduceBatchFuse : public OpBatchFuse {
 public:
  ReduceBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    context_->UpdateInitPad(input_tensor, {0, 0, 0, 0});
    context_->UpdateForwardPad(input_tensor, {0, 0, 0, 0});
    context_->UpdatePadInferShape(output_tensor, output_shape);
    context_->UpdateForwardPad(output_tensor, {0, 0, 0, 0});
    next_tensors.push_back(output_tensor);
    return false;
  }
  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    context_->UpdateBackwardPad(input_tensor, {0, 0, 0, 0});
    // context_->UpdatePadInferShape(output_tensor, output_shape);
    return false;
  }
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensor) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    std::shared_ptr<vx::Tensor> slice_and_concat_out;

    uint32_t batch = input_batch_fuse_tensor->GetShape()[3];
    uint32_t batch_src = input_tensor->GetShape()[3];
    if (batch == 1 && batch_src != 1) {
      //insert slice and concat
      slice_and_concat_out =
          InsertSliceAndConcat(input_batch_fuse_tensor, true, input_tensor);
    } else {
      slice_and_concat_out = input_batch_fuse_tensor;
    }
    auto slice_and_concat_shape = slice_and_concat_out->GetShape();
    context_->UpdateTensorMap(input_tensor, slice_and_concat_out);
    context_->UpdateTensorBatchFuseMap(slice_and_concat_out, input_tensor);

    std::vector<int32_t> new_axis;
    for (uint32_t i = 0; i < op_->impl()->node()->nn_param.reduce.axis_num;
         ++i) {
      int32_t axis = op_->impl()->node()->nn_param.reduce.axis[i];
      if (axis < 0) axis += input_shape.size();
      new_axis.push_back(axis);
    }

    auto reduce = context_->batch_fuse_graph_->CreateOperation<OpType>(
        new_axis, op_->impl()->node()->nn_param.reduce.keep_dim);

    auto out_de_batch_fuse = CreateOutputsTensor();
    auto out_de_batch_shape = out_de_batch_fuse[0]->GetShape();

    (*reduce).BindInput(slice_and_concat_out).BindOutput(out_de_batch_fuse[0]);

    next_tensor.push_back(op_->impl()->OutputsTensor()[0]);
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