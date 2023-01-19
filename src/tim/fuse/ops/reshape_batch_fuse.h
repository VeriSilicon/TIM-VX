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
// #include "permute_vector.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {
class ReshapeBatchFuse : public OpBatchFuse {
 public:
  ReshapeBatchFuse(const std::shared_ptr<vx::Operation> op,
                   std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto i_src = op_->impl()->InputsTensor()[0];
    auto i_src_shape = i_src->GetShape();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto o_src_shape = o_src->GetShape();
    auto pad = context_->GetForwardPad(i_src);
    auto i_infer_shape = context_->GetPadInferShape(i_src);

    // same as input
    context_->UpdateInitPad(o_src, {0, 0, 0, 0});
    context_->UpdateForwardPad(o_src, pad);
    context_->UpdatePadInferShape(o_src, o_src_shape);
    next_tensors.push_back(o_src);
    return false;
  }

  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    return false;
  }

#define CREATE_AND_CONCAT_OP(idx, start, length)                             \
  \               
    auto idx##_op =                                                          \
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Slice>(0, start, \
                                                                   length);  \
  vx::ShapeType idx##_shape({out_channel, out_w, out_h, 1});                 \
  auto idx##_spec = i_src_spec.SetShape(idx##_shape);                  \
  auto idx##_tensor = context_->batch_fuse_graph_->CreateTensor(idx##_spec); \
  (*idx##_op).BindInput(i_src_batch_fuse).BindOutput(idx##_tensor);          \
  tensors.push_back(idx##_tensor);

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensor) override {
    auto i_src = op_->impl()->InputsTensor()[0];
    auto i_src_shape = i_src->GetShape();
    auto i_src_spec = i_src->GetSpec();
    auto i_src_batch_fuse = context_->GetMapedTensor(i_src);
    auto i_src_batch_fuse_shape = i_src_batch_fuse->GetShape();
    auto i_src_batch_fuse_spec = context_->GetMapedTensor(i_src)->GetSpec();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto o_src_sepc = o_src->GetSpec();
    auto reshape_op = op_->Clone(context_->batch_fuse_graph_);

    if (i_src_shape[3] != 1 && i_src_batch_fuse_shape[3] == 1) {
      if (o_src->GetSpec().attr_ == vx::TensorAttribute::OUTPUT) {
        uint32_t batch = i_src_shape[3];
        uint32_t batch_factor_w = ClosestFactors(i_src_shape[3]).first;
        uint32_t batch_factor_h = ClosestFactors(i_src_shape[3]).second;
        // uint32_t sqrt_batch = sqrt(batch);
        uint32_t batch_out_h = i_src_batch_fuse_shape[2];
        uint32_t batch_out_w = i_src_batch_fuse_shape[1];
        uint32_t out_h = i_src_shape[2];
        uint32_t out_w = i_src_shape[1];
        uint32_t out_channel = i_src_shape[0];

        //if there has shared pad between valid value, overlap size may be negative
        int32_t overlap_h = 0;
        int32_t overlap_w = 0;
        if (batch_factor_h - 1 == 0)
          overlap_h = 0;
        else
          overlap_h =
              (batch_out_h - batch_factor_h * out_h) / (batch_factor_h - 1);
        if (batch_factor_w - 1 == 0)
          overlap_w = 0;
        else
          overlap_w =
              (batch_out_w - batch_factor_w * out_w) / (batch_factor_w - 1);

        int32_t out_w_ = static_cast<int32_t>(out_w);
        int32_t out_h_ = static_cast<int32_t>(out_h);
        int32_t out_channel_ = static_cast<int32_t>(out_channel);

        std::vector<int32_t> axis_point_h(batch_factor_h, 0);
        std::vector<int32_t> axis_point_w(batch_factor_w, 0);

        std::vector<int32_t> length = {out_channel_, out_w_, out_h_, 1};
        std::vector<std::vector<int32_t>> start_point;

        for (int i = 0; i < batch_factor_h; i++) {
          axis_point_h[i] = 0 + i * (overlap_h + out_h);
        }

        for (int i = 0; i < batch_factor_w; i++) {
          axis_point_w[i] = 0 + i * (overlap_w + out_w);
        }

        for (int i = 0; i < batch_factor_w; i++) {
          for (int j = 0; j < batch_factor_h; j++) {
            start_point.push_back({0, axis_point_w[j], axis_point_h[i], 0});
          }
        }

        std::vector<std::shared_ptr<vx::Tensor>> tensors;
        for (int i = 0; i < batch; i++) {
          CREATE_AND_CONCAT_OP(i, start_point[i], length);
        }
        auto slice_shape = tensors[0]->GetSpec();

        auto concat =
            context_->batch_fuse_graph_->CreateOperation<vx::ops::Concat>(
                3, batch);
        auto concat_tensor =
            context_->batch_fuse_graph_->CreateTensor(i_src_spec);
        auto concat_shape = concat_tensor->GetShape();
        (*concat).BindInputs(tensors).BindOutput(concat_tensor);
        context_->UpdateTensorMap(i_src, concat_tensor);
        context_->UpdateTensorBatchFuseMap(concat_tensor, i_src);
      }
    }

    auto out = context_->batch_fuse_graph_->CreateTensor(o_src_sepc);
    context_->UpdateTensorMap(o_src, out);
    context_->UpdateTensorBatchFuseMap(out, o_src);
    (*reshape_op).BindInput(context_->GetMapedTensor(i_src)).BindOutput(out);
    next_tensor.push_back(o_src);
  }
};
}  // namespace fuse
}  // namespace tim
#endif