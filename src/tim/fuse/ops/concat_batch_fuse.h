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
#ifndef TIM_BATCH_FUSE_CONCAT_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_CONCAT_BATCH_FUSE_H_

#include "tim/vx/ops/concat.h"

#include "op_batch_fuse.h"
// #include "permute_vector.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {
class ConcatBatchFuse : public OpBatchFuse {
 public:
  ConcatBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto o_src_shape = o_src->GetShape();
    auto i_src = op_->impl()->InputsTensor();
    auto i_src_map_0 = context_->GetPadInferShape(i_src[0]);
    auto i_src_map_1 = context_->GetPadInferShape(i_src[1]);

    uint32_t batch = o_src_shape[3];
    if (i_src_map_0[0] != i_src_map_1[0] && i_src_map_0[1] != i_src_map_1[1] && i_src_map_0[3] != i_src_map_1[3]){
      if (i_src_map_0[3] == 1) {
        auto i_src_gap_0 = context_->GetForwardGap(i_src[0]);
        context_->UpdateForwardGap(o_src, i_src_gap_0);
        
        context_->UpdatePadInferShape(o_src, {i_src_map_0[0], i_src_map_0[1], i_src_map_0[2] + i_src_map_1[2], i_src_map_0[3]});
      }
      if (i_src_map_1[3] == 1) {
        auto i_src_gap_1 = context_->GetForwardGap(i_src[1]);
        context_->UpdateForwardGap(o_src, i_src_gap_1);
        context_->UpdatePadInferShape(o_src, {i_src_map_1[0], i_src_map_1[1], i_src_map_0[2] + i_src_map_1[2], i_src_map_1[3]});
      }
    } else {
      auto i_src_gap_0 = context_->GetForwardGap(i_src[0]);
      context_->UpdateForwardGap(o_src, i_src_gap_0);
      context_->UpdatePadInferShape(o_src,{i_src_map_0[0], i_src_map_0[1], i_src_map_0[2] + i_src_map_1[2], i_src_map_0[3]});
    }

    context_->UpdateInitPad(i_src[0], {0, 0, 0, 0});
    context_->UpdateInitPad(i_src[1], {0, 0, 0, 0});
    context_->UpdateForwardPad(i_src[0], context_->GetForwardPad(i_src[0]));
    context_->UpdateForwardPad(i_src[1], context_->GetForwardPad(i_src[1]));

    next_tensors.push_back(o_src);
    context_->UpdateForwardPad(o_src, {0, 0, 0, 0});

    return false;
  }

  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto i_src = op_->impl()->InputsTensor();
    auto i_src_map_0 = context_->GetPadInferShape(i_src[0]);
    auto i_src_map_1 = context_->GetPadInferShape(i_src[1]);
    // auto map_shape_0 = i_src_map_0->GetShape();
    // auto map_shape_1 = i_src_map_1->GetShape();
    if (i_src_map_0[0] == i_src_map_1[0] && i_src_map_0[1] == i_src_map_1[1] && i_src_map_0[3] == i_src_map_1[3]) {
      // continue to backward
      former_tensors.push_back(i_src[0]);
      former_tensors.push_back(i_src[1]);
      auto out_shape = context_->GetPadInferShape(o_src);
      context_->UpdatePadInferShape(i_src[0],
                                    {out_shape[0], out_shape[1], i_src_map_0[2], out_shape[3]});
      context_->UpdatePadInferShape(i_src[1],
                                    {out_shape[0], out_shape[1], i_src_map_1[2], out_shape[3]});
      auto gap_1 = context_->GetForwardGap(i_src[0]);
      auto gap_2 = context_->GetForwardGap(i_src[1]);
      auto gap = context_->GetForwardGap(o_src);
      if (gap_1 == gap_2){
          context_->UpdateForwardGap(i_src[0], gap);
          context_->UpdateForwardGap(i_src[1], gap);
      }
      return true;
    }
    //else backward break

    return false;
  }
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto o_src = op_->impl()->OutputsTensor()[0];

    auto i_src = op_->impl()->InputsTensor();
    auto i_src_map_0 = context_->GetMapedTensor(i_src[0]);
    auto i_src_map_1 = context_->GetMapedTensor(i_src[1]);
    auto map_shape_0 = i_src_map_0->GetShape();
    auto map_shape_1 = i_src_map_1->GetShape();
    if (map_shape_0 != map_shape_1) {
      if (map_shape_0[3] != 1) {
        auto pad_tensor = InsertPad(i_src_map_0, false, i_src_map_0);
        auto batch_fuse_tensor_0 =
            InsertPermuteAndReshape(pad_tensor, false, i_src_map_0);
        context_->UpdateTensorMap(i_src[0], batch_fuse_tensor_0);
        map_shape_0 = batch_fuse_tensor_0->GetShape();
      }
      if (map_shape_1[3] != 1) {
        auto pad_tensor = InsertPad(i_src_map_1, false, i_src_map_1);
        auto batch_fuse_tensor_1 =
            InsertPermuteAndReshape(pad_tensor, false, i_src_map_1);
        context_->UpdateTensorMap(i_src[1], batch_fuse_tensor_1);
        map_shape_1 = batch_fuse_tensor_1->GetShape();
      }
    }

    auto axis = op_->impl()->node()->nn_param.concat.axis;
    auto concat = context_->batch_fuse_graph_->CreateOperation<vx::ops::Concat>(
        axis, op_->impl()->InputsTensor().size());
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*concat).BindInput(context_->GetMapedTensor(i_src));
    }

    vx::ShapeType out_shape;
    for (uint32_t i = 0; i < map_shape_0.size(); ++i) {
      if (i == axis) {
        out_shape.push_back(map_shape_0[i] + map_shape_1[i]);
      } else {
        out_shape.push_back(map_shape_0[i]);
      }
    }

    auto o_src_spec = o_src->GetSpec();
    auto out_spec = o_src_spec.SetShape(out_shape);
    // vx::TensorSpec out_spec(o_src->GetSpec().datatype_, out_shape,
    // o_src->GetSpec().attr_);

    auto out_concat = context_->batch_fuse_graph_->CreateTensor(out_spec);

    (*concat).BindOutput(out_concat);
    context_->UpdateTensorMap(o_src, out_concat);
    context_->UpdateTensorBatchFuseMap(out_concat, o_src);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace fuse
}  // namespace tim
#endif