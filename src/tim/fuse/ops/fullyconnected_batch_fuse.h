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
#ifndef TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_

#include "tim/vx/ops/conv2d.h"

#include "builtin_op_impl.h"
// #include "permute_vector.h"
#include "op_batch_fuse.h"

namespace tim {
namespace fuse {
class FullyConnectedBatchFuse : public OpBatchFuse {
 public:
  FullyConnectedBatchFuse(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}
  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {return false;}
  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {return false;}
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensor = input_tensors[0];
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    std::shared_ptr<vx::Tensor> slice_and_concat_out;
    // std::shared_ptr<vx::Tensor> fc_out_tensor;

    //input_tesnor
    uint32_t batch = context_->GetMapedTensor(input_tensor)->GetShape()[3];
    uint32_t batch_src = input_tensor->GetShape()[3];
    if (batch == 1 && batch_src != 1) {
      //insert slice and concat
      slice_and_concat_out =
          InsertSliceAndConcat(input_batch_fuse_tensor, true, input_tensor);
    } else {
      slice_and_concat_out = input_batch_fuse_tensor;
    }
    context_->UpdateTensorMap(input_tensor, slice_and_concat_out);
    context_->UpdateTensorBatchFuseMap(slice_and_concat_out, input_tensor);

    auto fcl = op_->Clone(context_->batch_fuse_graph_);
    auto out_de_batch_fuse = CreateOutputsTensor();
    for (auto in : op_->impl()->InputsTensor()) {
      (*fcl).BindInput(context_->GetMapedTensor(in));
    }

    (*fcl).BindOutput(out_de_batch_fuse[0]);
    // context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace fuse
}  // namespace tim

#endif