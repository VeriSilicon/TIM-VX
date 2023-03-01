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
#ifndef TIM_BATCH_FUSE_ElEMENTWISE_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_ElEMENTWISE_BATCH_FUSE_H_

#include "tim/vx/ops/elementwise.h"
#include "op_batch_fuse.h"
#include "builtin_op_impl.h"

namespace tim {
namespace fuse {
template <typename OpType>
class ElementWiseBatchFuse : public OpBatchFuse {
 public:
  ElementWiseBatchFuse() {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
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

    uint32_t batch = output_shape[batch_axis_0];

    //We assume two tensors' shape are equal after batch fuse
    //TODO(HuanyuCai): if two tensors are not equal after batch fuse, do not fuse them
    if (input_gap_infer_shape_0 == input_gap_infer_shape_1) {
      //Elementwise op not affect output shape
      context_->UpdateGapInferShape(output_tensor, input_gap_infer_shape_1);
    } else {
      //When batch fuse, InsertSliceandConcat need to be inserted before this op , so the input bacth != 1
      //so forward and backward are broken
      //the output of this op is like a graph input which need insert pad after forward and backward
      //then it need insert tanspose to batch fuse
      if (batch != 1) {
        //Batch fuse
        uint32_t batch_factor_w = ClosestFactors(batch).first;
        uint32_t batch_factor_h = ClosestFactors(batch).second;
        auto output_batch_fuse_w = batch_factor_w * output_shape[w_axis_0];
        auto output_batch_fuse_h = batch_factor_h * output_shape[h_axis_0];

        vx::ShapeType output_batch_fuse_shape(4, 0);
        output_batch_fuse_shape[w_axis_0] = output_batch_fuse_w;
        output_batch_fuse_shape[h_axis_0] = output_batch_fuse_h;
        output_batch_fuse_shape[c_axis_0] = output_shape[c_axis_0];
        output_batch_fuse_shape[batch_axis_0] = 1;

        context_->UpdateGapInferShape(output_tensor, output_batch_fuse_shape);
      } else {
        context_->UpdateGapInferShape(output_tensor, output_shape);
      }
    }
    next_tensors.push_back(output_tensor);
    //Output's gap is equal to input's for we assume two branches are balanced
    auto output_gap = context_->GetForwardGap(input_tensors[0]);
    context_->UpdateForwardGap(output_tensor, output_gap);

    //Elementwise op do not need to backward, return false always
    return false;
  }
  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_gap_infer_shape_0 = context_->GetGapInferShape(input_tensors[0]);
    auto input_gap_infer_shape_1 = context_->GetGapInferShape(input_tensors[1]);

    if (input_gap_infer_shape_0 == input_gap_infer_shape_1) {
      //Continue to backward because elementwis op do not affect the shape
      former_tensors.push_back(input_tensors[0]);
      former_tensors.push_back(input_tensors[1]);
      context_->UpdateGapInferShape(input_tensors[0],
                                    context_->GetGapInferShape(output_tensor));
      context_->UpdateGapInferShape(input_tensors[1],
                                    context_->GetGapInferShape(output_tensor));
      auto input_gap_0 = context_->GetForwardGap(input_tensors[0]);
      auto input_gap_1 = context_->GetForwardGap(input_tensors[1]);
      auto output_gap = context_->GetForwardGap(output_tensor);
      if (input_gap_0 == input_gap_1) {
        //If the are balanced

        context_->UpdateForwardGap(input_tensors[0], output_gap);
        context_->UpdateForwardGap(input_tensors[1], output_gap);
      } else {
        //TODO(HuanyuCai): if they are not balanced
        VSILOGW("Unbalanced branches are not supported yet");
      }
      return true;
    } else {
      //TODO(HuanyuCai): one tensor is not batch fused tensor shape and it need fuse.
      //If they are both fused and cannot be balanced, backward stop
    }

    return false;
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensors = op_->impl()->InputsTensor();
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

    std::shared_ptr<vx::Tensor> ele_out;
    auto output_spec = output_tensor->GetSpec();
    auto input_shape_0 = input_tensors[0]->GetShape();
    auto input_shape_1 = input_tensors[0]->GetShape();

    if (input_batch_fuse_shape_0 == input_batch_fuse_shape_1) {
      auto ele_out_spec = output_spec.SetShape(input_batch_fuse_shape_0);
      ele_out = context_->GetBatchFuseGraph()->CreateTensor(ele_out_spec);
      auto valid_prop =
          (float)(input_shape_0[w_axis_0] * input_shape_0[h_axis_0] *
                  input_shape_0[c_axis_0]) /
          (float)(input_batch_fuse_shape_0[w_axis_0] *
                  input_batch_fuse_shape_0[h_axis_0]);
      context_->UpdateProportion(input_tensors[0], valid_prop);
      context_->UpdateProportion(input_tensors[1], valid_prop);
    } else {
      //Two tensors' shape are not equal, so need slice and concat to be original tensors
      uint32_t batch = input_batch_fuse_shape_0[batch_axis_0];
      uint32_t batch_src = input_shape_0[batch_axis_0];
      std::shared_ptr<vx::Tensor> slice_and_concat_out_0;
      std::shared_ptr<vx::Tensor> slice_and_concat_out_1;

      if (batch == 1 && batch_src != 1) {
        //insert slice and concat
        slice_and_concat_out_0 =
            InsertSliceAndConcat(input_batch_fuse_tensor_0, input_tensors[0],
                                 batch_axis_0, fuse_axes_0);
        slice_and_concat_out_1 =
            InsertSliceAndConcat(input_batch_fuse_tensor_1, input_tensors[1],
                                 batch_axis_1, fuse_axes_1);
      } else {
        //Both tensors are fused tensor, n = 1
        slice_and_concat_out_0 = input_batch_fuse_tensor_0;
        slice_and_concat_out_1 = input_batch_fuse_tensor_0;
      }

      auto slice_and_concat_0_shape = slice_and_concat_out_0->GetShape();
      auto slice_and_concat_1_shape = slice_and_concat_out_1->GetShape();
      auto ele_out_spec = output_spec.SetShape(slice_and_concat_0_shape);
      ele_out = context_->GetBatchFuseGraph()->CreateTensor(ele_out_spec);

      //No batch fuse and no gap, so no dirty data
      context_->UpdateProportion(input_tensors[0], 1);
      context_->UpdateProportion(input_tensors[1], 1);
      context_->UpdateTensorMap(input_tensors[0], slice_and_concat_out_0);
      context_->UpdateTensorMap(input_tensors[1], slice_and_concat_out_1);
    }
    auto elementwise = context_->GetBatchFuseGraph()->CreateOperation<OpType>();
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*elementwise).BindInput(context_->GetMapedTensor(i_src));
    }

    (*elementwise).BindOutput(ele_out);
    context_->UpdateTensorMap(output_tensor, ele_out);
    next_tensors.push_back(output_tensor);
  }
};

using AddBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Add>;
using SubBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Sub>;
using DivBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Div>;
using PowBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Pow>;
using MinimumBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Minimum>;
using MaximumBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Maximum>;

}  // namespace fuse
}  // namespace tim

#endif