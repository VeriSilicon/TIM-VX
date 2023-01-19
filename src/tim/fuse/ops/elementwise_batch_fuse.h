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
#ifndef TIM_BATCH_FUSE_ElEMENTWISE_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_ElEMENTWISE_BATCH_FUSE_H_

#include "tim/vx/ops/elementwise.h"

#include "op_batch_fuse.h"
// #include "permute_vector.h"
#include "builtin_op_impl.h"

namespace tim {
namespace fuse {
template <typename OpType>
class ElementWiseBatchFuse : public OpBatchFuse {
 public:
  ElementWiseBatchFuse(
      const std::shared_ptr<vx::Operation> op,
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
    // auto map_shape_0 = i_src_map_0->GetShape();
    // auto map_shape_1 = i_src_map_1->GetShape();
    context_->UpdateInitPad(i_src[0], {0, 0, 0, 0});
    context_->UpdateInitPad(i_src[1], {0, 0, 0, 0});
    context_->UpdateForwardPad(i_src[0], context_->GetForwardPad(i_src[0]));
    context_->UpdateForwardPad(i_src[1], context_->GetForwardPad(i_src[1]));

    if (i_src_map_0 == i_src_map_1) {
      context_->UpdatePadInferShape(o_src, i_src_map_0);
    } else {
      //when batch fuse, InsertSliceandConcat need to be inserted before this op , so the input bacth != 1
      //so forward and backward are broken
      //the output of this op is like a graph input which need insert pad after forward and backward
      //then it need insert tanspose to batch fuse
      if (batch != 1) {
        //batch fuse

        uint32_t batch_factor_w = ClosestFactors(batch).first;
        uint32_t batch_factor_h = ClosestFactors(batch).second;
        auto batch_fuse_w_old = batch_factor_w * o_src_shape[0];
        auto batch_fuse_h_old = batch_factor_h * o_src_shape[1];
        vx::ShapeType batch_fuse_shape_old = {
            batch_fuse_w_old, batch_fuse_h_old, o_src_shape[2], 1};
        context_->UpdatePadInferShape(o_src, batch_fuse_shape_old);
      } else {
        context_->UpdatePadInferShape(o_src, o_src->GetShape());
      }
    }
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
    if (i_src_map_0 == i_src_map_1) {
      // continue to backward
      former_tensors.push_back(i_src[0]);
      former_tensors.push_back(i_src[1]);
      context_->UpdatePadInferShape(i_src[0],
                                    context_->GetPadInferShape(o_src));
      context_->UpdatePadInferShape(i_src[1],
                                    context_->GetPadInferShape(o_src));
      return true;
    }
    //else backward break

    return false;
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    // auto required_pv = AlignPermuteVectorForElementWise();
    auto o_src = op_->impl()->OutputsTensor()[0];
    auto i_src = op_->impl()->InputsTensor();
    auto i_src_map_0 = context_->GetMapedTensor(i_src[0]);
    auto i_src_map_1 = context_->GetMapedTensor(i_src[1]);
    auto map_shape_0 = i_src_map_0->GetShape();
    auto map_shape_1 = i_src_map_1->GetShape();
    std::shared_ptr<vx::Tensor> ele_out;
    auto o_src_spec = o_src->GetSpec();

    if (map_shape_0[0] == map_shape_1[0] && map_shape_0[1] == map_shape_1[1] &&
        map_shape_0[2] == map_shape_1[2] && map_shape_0[3] == map_shape_1[3]) {
      auto ele_out_spec =  o_src_spec.SetShape(map_shape_0);
      ele_out =
          context_->batch_fuse_graph_->CreateTensor(ele_out_spec);
    } else {
      uint32_t batch = context_->GetMapedTensor(i_src[0])->GetShape()[3];
      uint32_t batch_src = i_src[0]->GetShape()[3];
      std::shared_ptr<vx::Tensor> slice_and_concat_out_0;
      std::shared_ptr<vx::Tensor> slice_and_concat_out_1;

      if (batch == 1 && batch_src != 1) {
        //insert slice and concat
        slice_and_concat_out_0 =
            InsertSliceAndConcat(i_src_map_0, false, i_src[0]);
        slice_and_concat_out_1 =
            InsertSliceAndConcat(i_src_map_1, false, i_src[1]);
      } else {
        slice_and_concat_out_0 = i_src_map_0;
        slice_and_concat_out_1 = i_src_map_1;
      }

      auto slice_and_concat_0_shape = slice_and_concat_out_0->GetShape();
      auto slice_and_concat_1_shape = slice_and_concat_out_1->GetShape();
      auto ele_out_spec =  o_src_spec.SetShape(slice_and_concat_0_shape);
      ele_out = context_->batch_fuse_graph_->CreateTensor(
          ele_out_spec);
      

      context_->UpdateTensorMap(i_src[0], slice_and_concat_out_0);
      context_->UpdateTensorMap(i_src[1], slice_and_concat_out_1);
      context_->UpdateTensorBatchFuseMap(slice_and_concat_out_0, i_src[0]);
      context_->UpdateTensorBatchFuseMap(slice_and_concat_out_1, i_src[1]);
    }
    auto elementwise = context_->batch_fuse_graph_->CreateOperation<OpType>();
    for (const auto& i_src : op_->impl()->InputsTensor()) {
      (*elementwise).BindInput(context_->GetMapedTensor(i_src));
    }

    (*elementwise).BindOutput(ele_out);
    context_->UpdateTensorMap(o_src, ele_out);
    context_->UpdateTensorBatchFuseMap(ele_out, o_src);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

// class MultiplyLayoutInfer : public OpLayoutInfer {
//  public:
//   MultiplyLayoutInfer(
//       const std::shared_ptr<vx::Operation> op,
//       std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
//       : OpLayoutInfer(op, context) {}

//   void OnInputs(
//       std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
//     auto required_pv = AlignPermuteVectorForElementWise();
//     auto multiply =
//         context_->infer_graph_->CreateOperation<tim::vx::ops::Multiply>(
//             op_->impl()->node()->nn_param.multiply.scale);
//     for (const auto& i_src : op_->impl()->InputsTensor()) {
//       (*multiply).BindInput(context_->GetMapedTensor(i_src));
//     }
//     auto out_infer = CreateOutputsTensor(required_pv);
//     (*multiply).BindOutput(out_infer[0]);
//     context_->SetPermuteVector(op_->impl()->OutputsTensor()[0], required_pv);
//     next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
//   }
// };

using AddBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Add>;
using SubBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Sub>;
using DivBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Div>;
using PowBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Pow>;
using MinimumBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Minimum>;
using MaximumBatchFuse = ElementWiseBatchFuse<tim::vx::ops::Maximum>;

}  // namespace fuse
}  // namespace tim

#endif