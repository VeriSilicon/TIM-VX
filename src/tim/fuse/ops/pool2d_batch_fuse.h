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
#ifndef TIM_BATCH_FUSE_POOL2D_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_POOL2D_BATCH_FUSE_H_

#include "op_batch_fuse.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/pool2d.h"

namespace tim {
namespace fuse {
class Pool2dBatchFuse : public OpBatchFuse {
 public:
  Pool2dBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto output_tensor = op_->impl()->OutputsTensor()[0];

    auto input_shape = input_tensor->GetShape();
    auto output_shape = output_tensor->GetShape();  //whcn

    // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    // auto batch_src_axis = context_->GetBatchAxis();  // 3
    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    uint32_t batch = input_shape[batch_axis];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.pool.ksize[0],
        op_->impl()->node()->nn_param.pool.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.pool.stride[0],
        op_->impl()->node()->nn_param.pool.stride[1]};
    auto pad_type =
        TranslatePadType(op_->impl()->node()->nn_param.pool.pad_type);
    std::array<uint32_t, 4> pad = {
        op_->impl()->node()->nn_param.pool.pad[0],
        op_->impl()->node()->nn_param.pool.pad[1],
        op_->impl()->node()->nn_param.pool.pad[2],
        op_->impl()->node()->nn_param.pool.pad[3]};  // {0, 0, 0, 0}

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME || pad_type == vx::PadType::VALID) {
        //Compute the positive and negative value of pad size
        int32_t p_w =
            stride[0] * output_shape[w_axis] - input_shape[w_axis] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[h_axis] - input_shape[h_axis] + ksize[1] - stride[1];

        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGE("AUTO pad is not supported yet");
      }
    }
    if (int_pad[0] > 0 || int_pad[1] > 0 || int_pad[2] > 0 || int_pad[3] > 0) {
      //Do not batch fuse and pad inside
      //Becasue we can not pad the right value before running time

      context_->UpdateGapInferShape(input_tensor, input_shape);
      context_->UpdateGapInferShape(output_tensor, output_shape);

      context_->UpdateForwardGap(input_tensor, {0, 0});
      context_->UpdateForwardGap(output_tensor, {0, 0});
      next_tensors.push_back(output_tensor);

      //Do not batch fuse, so do not backward
      return false;
    }

    auto input_batch_fuse_shape_old = context_->GetGapInferShape(input_tensor);

    auto input_batch_fuse_w_old = input_batch_fuse_shape_old[w_axis];
    auto input_batch_fuse_h_old = input_batch_fuse_shape_old[h_axis];
    if (input_batch_fuse_shape_old[batch_axis] != 1) {
      //batch fuse
      input_batch_fuse_w_old *= batch_factor_w;
      input_batch_fuse_h_old *= batch_factor_h;
      input_batch_fuse_shape_old[w_axis] = input_batch_fuse_w_old;
      input_batch_fuse_shape_old[h_axis] = input_batch_fuse_h_old;
      input_batch_fuse_shape_old[c_axis] = input_batch_fuse_shape_old[c_axis];
      input_batch_fuse_shape_old[batch_axis] = 1;

      context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_old);
    }

    std::array<uint32_t, 2> gap = {0, 0};
    //The derivation process of gap size is as same as conv2d's
    auto m_w = ceil((float_t)(ksize[0] - int_pad[0] - int_pad[1]) /
                    (float_t)stride[0]);
    auto m_h = ceil((float_t)(ksize[1] - int_pad[2] - int_pad[3]) /
                    (float_t)stride[1]);

    gap[0] = (m_w + 2) * stride[0] + int_pad[0] + int_pad[1] - ksize[0];
    gap[1] = (m_h + 2) * stride[1] + int_pad[2] + int_pad[3] - ksize[1];
    context_->UpdateForwardGap(input_tensor, gap);

    auto input_batch_fuse_w_new =
        input_shape[w_axis] * batch_factor_w + (batch_factor_w - 1) * gap[0];
    auto input_batch_fuse_h_new =
        input_shape[h_axis] * batch_factor_h + (batch_factor_h - 1) * gap[1];

    vx::ShapeType input_batch_fuse_shape_new(4);
    input_batch_fuse_shape_new[w_axis] = input_batch_fuse_w_new;
    input_batch_fuse_shape_new[h_axis] = input_batch_fuse_h_new;
    input_batch_fuse_shape_new[c_axis] = input_shape[c_axis];
    input_batch_fuse_shape_new[batch_axis] = 1;

    bool need_backward = false;
    uint32_t input_w, input_h;
    if (input_batch_fuse_w_new > input_batch_fuse_w_old ||
        input_batch_fuse_h_new > input_batch_fuse_h_old) {
      context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_new);
      need_backward = true;  //need backward to update pad
      input_w = input_batch_fuse_w_new;
      input_h = input_batch_fuse_h_new;

    } else {
      //old(may have been updated) tensor size > new(the smallest known) size, use old size as input
      input_w = input_batch_fuse_w_old;
      input_h = input_batch_fuse_h_old;
    }

    //cal the batch fused output shape of this conv2d
    uint32_t output_batch_fuse_w_update = 0;
    uint32_t output_batch_fuse_h_update = 0;
    output_batch_fuse_w_update =
        ceil((float_t)(input_w - ksize[0] + int_pad[0] + int_pad[1]) /
                 (float_t)(stride[0]) +
             1);
    output_batch_fuse_h_update =
        ceil((float_t)(input_h - ksize[1] + int_pad[2] + int_pad[3]) /
                 (float_t)(stride[1]) +
             1);

    vx::ShapeType input_batch_fuse_shape_update(4);
    input_batch_fuse_shape_update[w_axis] = input_w;
    input_batch_fuse_shape_update[h_axis] = input_h;
    input_batch_fuse_shape_update[c_axis] = input_shape[c_axis];
    input_batch_fuse_shape_update[batch_axis] = 1;

    context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_update);

    //update output tensor -> fused tensor shape with this temporary pad
    vx::ShapeType output_batch_fuse_shape_update(4);
    output_batch_fuse_shape_update[w_axis] = output_batch_fuse_w_update;
    output_batch_fuse_shape_update[h_axis] = output_batch_fuse_h_update;
    output_batch_fuse_shape_update[c_axis] = output_shape[c_axis];
    output_batch_fuse_shape_update[batch_axis] = 1;

    context_->UpdateGapInferShape(output_tensor,
                                  output_batch_fuse_shape_update);

    auto gap_output_w =
        (output_batch_fuse_w_update - output_shape[w_axis] * batch_factor_w) /
        (batch_factor_w - 1);
    auto gap_output_h =
        (output_batch_fuse_w_update - output_shape[h_axis] * batch_factor_w) /
        (batch_factor_w - 1);
    std::array<uint32_t, 2> output_gap = {gap_output_w, gap_output_h};
    context_->UpdateForwardGap(output_tensor, output_gap);

    next_tensors.push_back(output_tensor);
    return need_backward;
  }
  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();

    // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    // auto batch_src_axis = context_->GetBatchAxis();  // 3
    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    auto output_batch_fuse_shape = context_->GetGapInferShape(output_tensor);
    auto input_batch_fuse_shape = context_->GetGapInferShape(input_tensor);

    uint32_t batch = input_tensor->GetShape()[batch_axis];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.pool.ksize[0],
        op_->impl()->node()->nn_param.pool.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.pool.stride[0],
        op_->impl()->node()->nn_param.pool.stride[1]};
    auto pad_type =
        TranslatePadType(op_->impl()->node()->nn_param.pool.pad_type);
    std::array<uint32_t, 4> pad = {
        op_->impl()->node()->nn_param.pool.pad[0],
        op_->impl()->node()->nn_param.pool.pad[1],
        op_->impl()->node()->nn_param.pool.pad[2],
        op_->impl()->node()->nn_param.pool.pad[3]};  // {0, 0, 0, 0}

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME) {
        int32_t p_w =
            stride[0] * output_shape[w_axis] - input_shape[w_axis] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[h_axis] - input_shape[h_axis] + ksize[1] - stride[1];

        //Front pad >= back pad
        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGE("AUTO pad is not supported yet");
      }
    }

    if (int_pad[0] > 0 || int_pad[1] > 0 || int_pad[2] > 0 || int_pad[3] > 0) {
      //Do not batch fuse and pad inside
      //Becasue we can not pad the right value before running time
      context_->UpdateForwardGap(output_tensor,
                                 context_->GetForwardGap(output_tensor));
      context_->UpdateGapInferShape(output_tensor,
                                    context_->GetGapInferShape(output_tensor));
      return false;
    }

    std::array<uint32_t, 2> gap_input = {0, 0};
    std::array<uint32_t, 2> gap_output = context_->GetForwardGap(output_tensor);
    //The derivation process of gap size is as same as conv2d's
    gap_input[0] =
        stride[0] * (gap_output[0] + 1) + int_pad[0] + int_pad[1] - 1;
    gap_input[1] =
        stride[1] * (gap_output[1] + 1) + int_pad[2] + int_pad[3] - 1;
    context_->UpdateForwardGap(input_tensor, gap_input);

    uint32_t input_batch_fuse_w_update =
        input_shape[w_axis] * batch_factor_w + (batch_factor_w - 1) * gap_input[0];

    uint32_t input_batch_fuse_h_update =
        input_shape[h_axis] * batch_factor_h + (batch_factor_h - 1) * gap_input[1];

    bool need_backward = false;
    if ((input_batch_fuse_shape[w_axis] < input_batch_fuse_w_update) ||
        (input_batch_fuse_shape[h_axis] < input_batch_fuse_h_update)) {
      //Continue backward
      vx::ShapeType input_batch_fuse_shape_update(4);
      input_batch_fuse_shape_update[w_axis] = input_batch_fuse_w_update;
      input_batch_fuse_shape_update[h_axis] = input_batch_fuse_h_update;
      input_batch_fuse_shape_update[c_axis] = input_shape[c_axis];
      input_batch_fuse_shape_update[batch_axis] = 1;

      former_tensors.push_back(input_tensor);
      need_backward = true;
      context_->UpdateGapInferShape(input_tensor,
                                    input_batch_fuse_shape_update);
    }

    //Computes the new output size with updated input for updated output size may bigger than the smallest known size
    uint32_t output_batch_fuse_w_update = 0;
    uint32_t output_batch_fuse_h_update = 0;

    output_batch_fuse_w_update =
        ceil((float_t)(input_batch_fuse_w_update + int_pad[0] + int_pad[1] -
                       ksize[0]) /
             (float_t)(stride[0])) +
        1;
    output_batch_fuse_h_update =
        ceil((float_t)(input_batch_fuse_h_update + int_pad[2] + int_pad[3] -
                       ksize[1]) /
             (float_t)(stride[1])) +
        1;

    vx::ShapeType output_batch_fuse_shape_update(4);
    output_batch_fuse_shape_update[w_axis] = output_batch_fuse_w_update;
    output_batch_fuse_shape_update[h_axis] = output_batch_fuse_h_update;
    output_batch_fuse_shape_update[c_axis] = output_shape[c_axis];
    output_batch_fuse_shape_update[batch_axis] = 1;

    //Update tensor shape map
    context_->UpdateGapInferShape(output_tensor,
                                  output_batch_fuse_shape_update);
    return need_backward;  // useless
  }
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_spec = input_tensor->GetSpec();
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
     // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    // auto batch_src_axis = context_->GetBatchAxis();  // 3
    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    auto batch_src = input_shape[batch_axis];
    auto batch = input_batch_fuse_shape[batch_axis];
    auto channel = input_batch_fuse_shape[c_axis];

    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.pool.ksize[0],
        op_->impl()->node()->nn_param.pool.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.pool.stride[0],
        op_->impl()->node()->nn_param.pool.stride[1]};
    auto pool_type = TranslatePoolType(op_->impl()->node()->nn_param.pool.type);
    auto round_type =
        TranslateRoundType(op_->impl()->node()->nn_param.pool.round_type);
    auto pad_type =
        TranslatePadType(op_->impl()->node()->nn_param.pool.pad_type);
    std::array<uint32_t, 4> pad = {
        op_->impl()->node()->nn_param.pool.pad[0],
        op_->impl()->node()->nn_param.pool.pad[1],
        op_->impl()->node()->nn_param.pool.pad[2],
        op_->impl()->node()->nn_param.pool.pad[3]};  // {0, 0, 0, 0}

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME) {
        
        int32_t p_w =
            stride[0] * output_shape[w_axis] - input_shape[w_axis] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[h_axis] - input_shape[h_axis] + ksize[1] - stride[1];

        
        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGE("AUTO pad is not supported yet");
      }
    }

    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    std::shared_ptr<vx::Tensor> pool2d_out_tensor;
    uint32_t out_w_batch = 0;
    uint32_t out_h_batch = 0;

    if (batch != 1) {
      if (int_pad[0] <= 0 && int_pad[1] <= 0 && int_pad[2] <= 0 &&
          int_pad[3] <= 0)
      {
        //Pool wont be effected by pad value
        //Input tensor has not been batch fused, fuse it

       //step one, if it need pad, pad it with input tensor's gap infer shape
        auto pad_tensor = InsertPad(input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);

        //step two, batch fuse
        batch_fuse_tensor = InsertPermuteAndReshape(pad_tensor, input_tensor, batch_axis, fuse_axes);
      } else {
        //Do not batch fuse
        batch_fuse_tensor = input_tensor;
      }
    } else {
      //Has been fused before
      if (int_pad[0] <= 0 && int_pad[1] <= 0 && int_pad[2] <= 0 &&
          int_pad[3] <= 0) {
        batch_fuse_tensor = input_batch_fuse_tensor;
      } else {
        //Do not batch fuse
        batch_fuse_tensor =
            InsertSliceAndConcat(input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
      }
    }
    auto batch_fuse_shape = batch_fuse_tensor->GetShape();
    auto valid_prop = (float)(input_shape[w_axis] * input_shape[h_axis] * batch_src) /
                       (float)(batch_fuse_shape[w_axis] * batch_fuse_shape[h_axis]);
    context_->UpdateProportion(input_tensor, valid_prop);

    out_w_batch = ceil((float_t)(batch_fuse_shape[w_axis] - ksize[0] + int_pad[0] +
                                 int_pad[1]) /
                       (float_t)(stride[0])) +
                  1;
    out_h_batch = ceil((float_t)(batch_fuse_shape[h_axis] - ksize[1] + int_pad[2] +
                                 int_pad[3]) /
                       (float_t)(stride[1])) +
                  1;

    tim::vx::ShapeType pool_output_shape(4);
    pool_output_shape[w_axis] = out_w_batch;
    pool_output_shape[h_axis] = out_h_batch;
    pool_output_shape[c_axis] = channel;
    pool_output_shape[batch_axis] = batch_fuse_shape[batch_axis];

    auto output_spec = output_tensor->GetSpec();
    auto pool_output_spec = output_spec.SetShape(pool_output_shape);
    pool2d_out_tensor =
        context_->batch_fuse_graph_->CreateTensor(pool_output_spec);
    context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    context_->UpdateTensorMap(output_tensor, pool2d_out_tensor);

    //Insert mask to clean gap inside's value to be 0
    if (batch_fuse_tensor->GetShape()[batch_axis] == 1 && batch_src != 1) {
      auto masked_input = InsertMask(batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
      context_->UpdateTensorMap(input_tensor, masked_input);
    }

    auto pool2d = context_->batch_fuse_graph_->CreateOperation<vx::ops::Pool2d>(
        pool_type, pad_type, ksize, stride, round_type, vx::DataLayout::WHCN);

    (*pool2d).BindInput(context_->GetMapedTensor(input_tensor));
    (*pool2d).BindOutput(pool2d_out_tensor);

    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(output_tensor);
  }
};

}  // namespace fuse
}  // namespace tim

#endif