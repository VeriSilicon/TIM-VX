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
#ifndef TIM_BATCH_FUSE_CONCAT_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_CONCAT_BATCH_FUSE_H_

#include "tim/vx/ops/concat.h"
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"
namespace tim {
namespace fuse {
class ConcatBatchFuse : public OpBatchFuse {
 public:
  ConcatBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_gap_infer_shape_0 = context_->GetGapInferShape(input_tensors[0]);
    auto input_gap_infer_shape_1 = context_->GetGapInferShape(input_tensors[1]);

    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map_0 = context_->GetPermAxisMap(input_tensors[0]);
    auto fuse_axes_0 = context_->GetPermFuseAxes(input_tensors[0]);
    auto batch_axis_0 = context_->GetPermBatchAxis(input_tensors[0]);
    auto c_axis_0 = context_->GetPermChannelAxis(input_tensors[0]);
    auto w_axis_0 = fuse_axes_0[0];
    auto h_axis_0 = fuse_axes_0[1];

    auto perm_axis_map_1 = context_->GetPermAxisMap(input_tensors[1]);
    auto fuse_axes_1 = context_->GetPermFuseAxes(input_tensors[1]);
    auto batch_axis_1 = context_->GetPermBatchAxis(input_tensors[1]);
    auto c_axis_1 = context_->GetPermChannelAxis(input_tensors[1]);
    auto w_axis_1 = fuse_axes_0[0];
    auto h_axis_1 = fuse_axes_0[1];

    //TODO(HuanyuCai): did not find concat axis, default concat on channel axis
    if (input_gap_infer_shape_0[w_axis_0] !=
            input_gap_infer_shape_1[w_axis_1] &&
        input_gap_infer_shape_0[h_axis_0] !=
            input_gap_infer_shape_1[h_axis_1] &&
        input_gap_infer_shape_0[batch_axis_0] !=
            input_gap_infer_shape_1[batch_axis_1]) {
      //one input is not batch fused, but another one is batch fuesed
      if (input_gap_infer_shape_0[batch_axis_0] == 1) {
        //input tensor 0 is batch fused, use its gap to initialize output's gap
        auto input_gap = context_->GetForwardGap(input_tensors[0]);
        context_->UpdateForwardGap(output_tensor, input_gap);

        //concat on channel axis, set new shape after concat
        vx::ShapeType tmp_shape(4, 0);
        tmp_shape[w_axis_0] = input_gap_infer_shape_0[w_axis_0];
        tmp_shape[h_axis_0] = input_gap_infer_shape_0[h_axis_0];
        tmp_shape[c_axis_0] = input_gap_infer_shape_0[c_axis_0] +
                              input_gap_infer_shape_1[c_axis_1];
        tmp_shape[batch_axis_0] = input_gap_infer_shape_0[batch_axis_1];
        context_->UpdateGapInferShape(output_tensor, tmp_shape);
      }
      if (input_gap_infer_shape_1[batch_axis_1] == 1) {
        auto input_gap = context_->GetForwardGap(input_tensors[1]);
        context_->UpdateForwardGap(output_tensor, input_gap);

        vx::ShapeType tmp_shape(4, 0);
        tmp_shape[w_axis_1] = input_gap_infer_shape_1[w_axis_1];
        tmp_shape[h_axis_1] = input_gap_infer_shape_1[h_axis_1];
        tmp_shape[c_axis_1] = input_gap_infer_shape_0[c_axis_0] +
                              input_gap_infer_shape_1[c_axis_1];
        tmp_shape[batch_axis_1] = input_gap_infer_shape_1[batch_axis_1];
        context_->UpdateGapInferShape(output_tensor, tmp_shape);
      }
    } else {
      //both tensor are batch fused or are not batch fused
      auto input_gap = context_->GetForwardGap(input_tensors[0]);
      context_->UpdateForwardGap(output_tensor, input_gap);

      //set new shape after concat
      vx::ShapeType tmp_shape(4, 0);
      tmp_shape[w_axis_1] = input_gap_infer_shape_1[w_axis_1];
      tmp_shape[h_axis_1] = input_gap_infer_shape_1[h_axis_1];
      tmp_shape[c_axis_1] =
          input_gap_infer_shape_0[c_axis_0] + input_gap_infer_shape_1[c_axis_1];
      tmp_shape[batch_axis_1] = input_gap_infer_shape_1[batch_axis_1];
      context_->UpdateGapInferShape(output_tensor, tmp_shape);
    }

    next_tensors.push_back(output_tensor);

    return false;
  }

  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_gap_infer_shape_0 = context_->GetGapInferShape(input_tensors[0]);
    auto input_gap_infer_shape_1 = context_->GetGapInferShape(input_tensors[1]);

    auto perm_axis_map_0 = context_->GetPermAxisMap(input_tensors[0]);
    auto fuse_axes_0 = context_->GetPermFuseAxes(input_tensors[0]);
    auto batch_axis_0 = context_->GetPermBatchAxis(input_tensors[0]);
    auto c_axis_0 = context_->GetPermChannelAxis(input_tensors[0]);
    auto w_axis_0 = fuse_axes_0[0];
    auto h_axis_0 = fuse_axes_0[1];

    auto perm_axis_map_1 = context_->GetPermAxisMap(input_tensors[1]);
    auto fuse_axes_1 = context_->GetPermFuseAxes(input_tensors[1]);
    auto batch_axis_1 = context_->GetPermBatchAxis(input_tensors[1]);
    auto c_axis_1 = context_->GetPermChannelAxis(input_tensors[1]);
    auto w_axis_1 = fuse_axes_0[0];
    auto h_axis_1 = fuse_axes_0[1];

    // TODO(HuanyuCai): did not find concat axis, default concat on channel axis
    if (input_gap_infer_shape_0[w_axis_0] ==
            input_gap_infer_shape_1[w_axis_1] &&
        input_gap_infer_shape_0[h_axis_0] ==
            input_gap_infer_shape_1[h_axis_1] &&
        input_gap_infer_shape_0[batch_axis_0] ==
            input_gap_infer_shape_1[batch_axis_1]) {
      // continue to backward
      former_tensors.push_back(input_tensors[0]);
      former_tensors.push_back(input_tensors[1]);

      auto output_gap_infer_shape = context_->GetGapInferShape(output_tensor);

      // in backward, update input tensors' pad infer shape with output tensor's w, h and n
      // keep their own c(channel)

      vx::ShapeType tmp_shape_0(4, 0);
      tmp_shape_0[w_axis_0] = output_gap_infer_shape[w_axis_0];
      tmp_shape_0[h_axis_0] = output_gap_infer_shape[h_axis_0];
      tmp_shape_0[c_axis_0] =
          input_gap_infer_shape_0[c_axis_0]; /*its own channel*/
      tmp_shape_0[batch_axis_0] = output_gap_infer_shape[batch_axis_0];
      context_->UpdateGapInferShape(input_tensors[0], tmp_shape_0);

      vx::ShapeType tmp_shape_1(4, 0);
      tmp_shape_1[w_axis_1] = output_gap_infer_shape[w_axis_1];
      tmp_shape_1[h_axis_1] = output_gap_infer_shape[h_axis_1];
      tmp_shape_1[c_axis_1] =
          input_gap_infer_shape_1[c_axis_1]; /*its own channel*/
      tmp_shape_1[batch_axis_1] = output_gap_infer_shape[batch_axis_1];
      context_->UpdateGapInferShape(input_tensors[1], tmp_shape_1);

      //update input tensors' gap with output tensor's gap
      auto input_gap_0 = context_->GetForwardGap(input_tensors[0]);
      auto input_gap_1 = context_->GetForwardGap(input_tensors[1]);
      auto output_gap = context_->GetForwardGap(output_tensor);
      if (input_gap_0 == input_gap_1) {
        context_->UpdateForwardGap(input_tensors[0], output_gap);
        context_->UpdateForwardGap(input_tensors[1], output_gap);
      } else {
        // TODO(HuanyuCai): Handle two branches' unbalance in both ForwardGapInference and BackwardGapInference
        VSILOGW("Unbalanced branches are not supported yet");
      }
      // it is balanced, so it can continue to backward
      return true;
    } 
    else {
      // unbalance
      VSILOGW("Unbalanced branches are not supported yet");
      return false;
    }
  }
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto output_tensor = op_->impl()->OutputsTensor()[0];

    auto input_tensors = op_->impl()->InputsTensor();
    auto input_shape_0 = input_tensors[0]->GetShape();
    auto input_shape_1 = input_tensors[1]->GetShape();
    auto input_batch_fuse_tensor_0 = context_->GetMapedTensor(input_tensors[0]);
    auto input_batch_fuse_tensor_1 = context_->GetMapedTensor(input_tensors[1]);
    auto input_batch_fuse_shape_0 = input_batch_fuse_tensor_0->GetShape();
    auto input_batch_fuse_shape_1 = input_batch_fuse_tensor_1->GetShape();

    auto perm_axis_map_0 = context_->GetPermAxisMap(input_tensors[0]);
    auto fuse_axes_0 = context_->GetPermFuseAxes(input_tensors[0]);
    auto batch_axis_0 = context_->GetPermBatchAxis(input_tensors[0]);
    auto c_axis_0 = context_->GetPermChannelAxis(input_tensors[0]);
    auto w_axis_0 = fuse_axes_0[0];
    auto h_axis_0 = fuse_axes_0[1];

    auto perm_axis_map_1 = context_->GetPermAxisMap(input_tensors[1]);
    auto fuse_axes_1 = context_->GetPermFuseAxes(input_tensors[1]);
    auto batch_axis_1 = context_->GetPermBatchAxis(input_tensors[1]);
    auto c_axis_1 = context_->GetPermChannelAxis(input_tensors[1]);
    auto w_axis_1 = fuse_axes_1[0];
    auto h_axis_1 = fuse_axes_1[1];

    if (input_batch_fuse_shape_0[batch_axis_0] !=
        input_batch_fuse_shape_1[batch_axis_1]) {
      // one tensor is fused, another is not, we need they to be both fused, then concat them
      // TODO(HuanyuCai): have not take unbalanced branches into consideration, in resnet they are balanced
      if (input_batch_fuse_shape_0[batch_axis_0] != 1) {
        auto pad_tensor = InsertPad(input_batch_fuse_tensor_0, input_tensors[0],
                                    batch_axis_0, fuse_axes_0);
        auto batch_fuse_tensor_0 = InsertPermuteAndReshape(
            pad_tensor, input_tensors[0], batch_axis_0, fuse_axes_0);
        context_->UpdateTensorMap(input_tensors[0], batch_fuse_tensor_0);
        input_batch_fuse_shape_0 = batch_fuse_tensor_0->GetShape();
      }
      if (input_batch_fuse_shape_1[batch_axis_1] != 1) {
        auto pad_tensor = InsertPad(input_batch_fuse_tensor_1, input_tensors[1],
                                    batch_axis_1, fuse_axes_1);
        auto batch_fuse_tensor_1 = InsertPermuteAndReshape(
            pad_tensor, input_tensors[0], batch_axis_1, fuse_axes_1);
        context_->UpdateTensorMap(input_tensors[1], batch_fuse_tensor_1);
        input_batch_fuse_shape_1 = batch_fuse_tensor_1->GetShape();
      }
    }

    auto valid_ratio_0 =
        (float)(input_shape_0[w_axis_0] * input_shape_0[h_axis_0] *
                input_shape_0[c_axis_0]) /
        (float)(input_batch_fuse_shape_0[w_axis_1] *
                input_batch_fuse_shape_0[h_axis_1]);
    auto valid_ratio_1 =
        (float)(input_shape_1[w_axis_1] * input_shape_1[h_axis_1] *
                input_shape_0[c_axis_1]) /
        (float)(input_batch_fuse_shape_1[w_axis_1] *
                input_batch_fuse_shape_1[h_axis_1]);
    context_->UpdateProportion(input_tensors[0], valid_ratio_0);
    context_->UpdateProportion(input_tensors[1], valid_ratio_1);

    auto axis = op_->impl()->node()->nn_param.concat.axis;
    auto concat = context_->batch_fuse_graph_->CreateOperation<vx::ops::Concat>(
        axis, op_->impl()->InputsTensor().size());
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*concat).BindInput(context_->GetMapedTensor(i_src));
    }

    vx::ShapeType output_shape;
    for (uint32_t i = 0; i < input_batch_fuse_shape_0.size(); ++i) {
      if (i == axis) {
        //find concat axis
        output_shape.push_back(input_batch_fuse_shape_0[i] +
                               input_batch_fuse_shape_1[i]);
      } else {
        output_shape.push_back(input_batch_fuse_shape_0[i]);
      }
    }

    auto output_spec = output_tensor->GetSpec();

    // set batch fused new output shape of concat
    output_spec.SetShape(output_shape);
    auto out_concat = context_->batch_fuse_graph_->CreateTensor(output_spec);

    (*concat).BindOutput(out_concat);
    context_->UpdateTensorMap(output_tensor, out_concat);

    next_tensors.push_back(output_tensor);
  }
};

}  // namespace fuse
}  // namespace tim
#endif