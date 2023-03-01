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
#ifndef TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_

#include "tim/vx/ops/conv2d.h"
#include "builtin_op_impl.h"
#include "op_batch_fuse.h"

namespace tim {
namespace fuse {
class Conv2dBatchFuse : public OpBatchFuse {
 public:
  Conv2dBatchFuse() {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensor = input_tensors[0];
    auto weight_tensor = input_tensors[1];
    auto input_shape = input_tensor->GetShape();
    auto output_shape = output_tensor->GetShape();  //whcn
    auto weight_shape = weight_tensor->GetShape();

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    auto fuse_src_axes = context_->GetFuseAxes();  // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    auto pad_type = OpBatchFuse::TranslatePadType(
        op_->impl()->node()->nn_param.conv2d.pad_type);
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.conv2d.stride[0],
        op_->impl()->node()->nn_param.conv2d.stride[1]};
    std::array<uint32_t, 4> pad = {op_->impl()->node()->nn_param.conv2d.pad[0],
                                   op_->impl()->node()->nn_param.conv2d.pad[1],
                                   op_->impl()->node()->nn_param.conv2d.pad[2],
                                   op_->impl()->node()->nn_param.conv2d.pad[3]};

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME || pad_type == vx::PadType::VALID) {
        //Compute the positive and negative value of pad size
        int32_t p_w = stride[0] * output_shape[w_axis] - input_shape[w_axis] +
                      weight_shape[0] - stride[0];
        int32_t p_h = stride[1] * output_shape[h_axis] - input_shape[h_axis] +
                      weight_shape[1] - stride[1];

        //Default: front pad size <= back pad size
        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGW("AUTO pad is not supported yet");
      }
    }

    //get the old gap infered shape of input tensor
    auto input_batch_fuse_shape_old = context_->GetGapInferShape(input_tensor);

    auto input_batch_fuse_w_old = input_batch_fuse_shape_old[w_axis];
    auto input_batch_fuse_h_old = input_batch_fuse_shape_old[h_axis];
    if (input_batch_fuse_shape_old[batch_axis] != 1) {
      //batch fuse and update fused shape
      input_batch_fuse_w_old *= batch_factor_w;
      input_batch_fuse_h_old *= batch_factor_h;
      input_batch_fuse_shape_old[w_axis] = input_batch_fuse_w_old;
      input_batch_fuse_shape_old[h_axis] = input_batch_fuse_h_old;
      input_batch_fuse_shape_old[c_axis] = input_batch_fuse_shape_old[c_axis];
      input_batch_fuse_shape_old[batch_axis] = 1;

      context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_old);
    }

    std::array<uint32_t, 2> gap = {0, 0};

    //// The derivation process of gap size:
    //// uint32_t start_w = input_shape[0] + pad[1] - weight_shape[0] + stride[0];
    //// uint32_t start_h = input_shape[1] + pad[3] - weight_shape[1] - stride[0];
    //// uint32_t end_w = input_shape[0] + gap[0] - pad[0];
    //// uint32_t end_h = input_shape[1] + gap[1] - pad[2];
    //// (end_w - start_w) % stride[0] = 0 -> the smallest gap[0], gap[1] is the same
    //// Take w as an example, we assume (end_w - start_w) / strid[0] = m_w, m_w is a integer
    //// that is: (gap[0] - pad[0] - pad[1] + weght_shape[0] - 2 * stride[0]) / stride[0] = m_w
    //// gap[0] = (m_w + 2) * stride[0]  + pad[0] + pad[1] - weight_shape[0], which is >= 0
    //// so we can get the smallest (m_w + 2)

    auto m_w = ceil((float_t)(weight_shape[0] - int_pad[0] - int_pad[1]) /
                    (float_t)stride[0]);
    auto m_h = ceil((float_t)(weight_shape[1] - int_pad[2] - int_pad[3]) /
                    (float_t)stride[1]);

    gap[0] =
        (m_w + 2) * stride[0] + int_pad[0] + int_pad[1] - weight_shape[0];  //w
    gap[1] =
        (m_h + 2) * stride[1] + int_pad[2] + int_pad[3] - weight_shape[1];  //h

    context_->UpdateForwardGap(input_tensor, gap);

    auto input_batch_fuse_w_new =
        input_shape[w_axis] * batch_factor_w + (batch_factor_w - 1) * gap[0];
    auto input_batch_fuse_h_new =
        input_shape[h_axis] * batch_factor_h + (batch_factor_h - 1) * gap[1];

    // new batch fused shape with gap inside
    vx::ShapeType input_batch_fuse_shape_new(4);
    input_batch_fuse_shape_new[w_axis] = input_batch_fuse_w_new;
    input_batch_fuse_shape_new[h_axis] = input_batch_fuse_h_new;
    input_batch_fuse_shape_new[c_axis] = input_shape[c_axis];
    input_batch_fuse_shape_new[batch_axis] = 1;

    bool need_backward = false;
    uint32_t input_w, input_h;
    if (input_batch_fuse_w_new > input_batch_fuse_w_old ||
        input_batch_fuse_h_new > input_batch_fuse_h_old) {
      //the old fused shape is smaller than new fused shape with enough gap

      context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_new);
      need_backward = true;  //need backward to update gap inside

      //batch fused tensor to conv2d
      input_w = input_batch_fuse_w_new;
      input_h = input_batch_fuse_h_new;

    } else {
      //old(may have been updated) tensor size > new(the smallest known) size, use old size as input
      input_w = input_batch_fuse_w_old;
      input_h = input_batch_fuse_h_old;
    }

    //compute the batch fused output shape of this conv2d
    uint32_t output_batch_fuse_w_update = 0;
    uint32_t output_batch_fuse_h_update = 0;

    //let floor() to discard extra pixels rather negative pad size
    int_pad[1] = int_pad[1] < 0 ? 0 : int_pad[1];
    int_pad[3] = int_pad[3] < 0 ? 0 : int_pad[3];
    output_batch_fuse_w_update =
        floor((float_t)(input_w - weight_shape[0] + int_pad[0] + int_pad[1]) /
                  (float_t)(stride[0]) +
              1);
    output_batch_fuse_h_update =
        floor((float_t)(input_h - weight_shape[1] + int_pad[2] + int_pad[3]) /
                  (float_t)(stride[1]) +
              1);

    //update output tensor -> fused tensor shape with this temporary pad
    vx::ShapeType output_batch_fuse_shape_update(4);
    output_batch_fuse_shape_update[w_axis] = output_batch_fuse_w_update;
    output_batch_fuse_shape_update[h_axis] = output_batch_fuse_h_update;
    output_batch_fuse_shape_update[c_axis] = output_shape[c_axis];
    output_batch_fuse_shape_update[batch_axis] = 1;

    context_->UpdateGapInferShape(output_tensor,
                                  output_batch_fuse_shape_update);

    //compute the output gap
    auto gap_output_w =
        (output_batch_fuse_w_update - output_shape[w_axis] * batch_factor_w) /
        (batch_factor_w - 1);
    auto gap_output_h =
        (output_batch_fuse_h_update - output_shape[h_axis] * batch_factor_h) /
        (batch_factor_h - 1);
    std::array<uint32_t, 2> gap_output = {gap_output_w, gap_output_h};
    context_->UpdateForwardGap(output_tensor, gap_output);

    next_tensors.push_back(output_tensor);
    return need_backward;
  }

  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_tensor = input_tensors[0];
    auto weight_tensor = input_tensors[1];
    auto weight_shape = weight_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();

    // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    auto fuse_src_axes = context_->GetFuseAxes();  // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    auto output_batch_fuse_shape = context_->GetGapInferShape(output_tensor);
    auto input_batch_fuse_shape = context_->GetGapInferShape(input_tensor);

    uint32_t batch = input_shape[batch_axis];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    auto pad_type = OpBatchFuse::TranslatePadType(
        op_->impl()->node()->nn_param.conv2d.pad_type);
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.conv2d.stride[0],
        op_->impl()->node()->nn_param.conv2d.stride[1]};
    std::array<uint32_t, 4> pad = {op_->impl()->node()->nn_param.conv2d.pad[0],
                                   op_->impl()->node()->nn_param.conv2d.pad[1],
                                   op_->impl()->node()->nn_param.conv2d.pad[2],
                                   op_->impl()->node()->nn_param.conv2d.pad[3]};

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME || pad_type == vx::PadType::VALID) {
        //compute the positive and negative value of pad size
        int32_t p_w = stride[0] * output_shape[w_axis] - input_shape[w_axis] +
                      weight_shape[0] - stride[0];
        int32_t p_h = stride[1] * output_shape[h_axis] - input_shape[h_axis] +
                      weight_shape[1] - stride[1];

        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGW("AUTO pad is not supported yet");
      }
    }

    std::array<uint32_t, 2> gap_input = {0, 0};
    std::array<uint32_t, 2> gap_output = context_->GetForwardGap(output_tensor);

    //// the derivation process of gap size of input from output gap size:
    //// uint32_t start_w = input_shape[0] + pad[1] - weight_shape[0] + stride[0];
    //// uint32_t start_h = input_shape[1] + pad[3] - weight_shape[1] - stride[0];
    //// uint32_t end_w = input_shape[0] + gap_input[0] - pad[0];
    //// uint32_t end_h = input_shape[1] + gap_input[1] - pad[2];
    //// take w as an example, end_w - start_w + 1 = gap_input[0] - pad[0] - pad[1] - 2*stride[0] + 1 + weight_shape[0]
    //// is the input data which willl produce dirty data
    //// so, (gap_input[0] - pad[0] - pad[1] - 2*stride[0] + 1 + weight_shape[0] - weight_shape[0] + 1 ) / stride[0] + 1 = gap_output[0]
    //// so we can get gap_input[0]:

    gap_input[0] =
        stride[0] * (gap_output[0] + 1) + int_pad[0] + int_pad[1] - 1;
    gap_input[1] =
        stride[1] * (gap_output[1] + 1) + int_pad[2] + int_pad[3] - 1;
    context_->UpdateForwardGap(input_tensor, gap_input);

    // update the batch fused input shape with new gap inside
    uint32_t input_batch_fuse_w_update = input_shape[w_axis] * batch_factor_w +
                                         (batch_factor_w - 1) * gap_input[0];

    uint32_t input_batch_fuse_h_update = input_shape[h_axis] * batch_factor_h +
                                         (batch_factor_h - 1) * gap_input[1];

    bool need_backward = false;
    if ((input_batch_fuse_shape[w_axis] < input_batch_fuse_w_update) ||
        (input_batch_fuse_shape[h_axis] < input_batch_fuse_h_update)) {
      //continue backward and update new batch fused input tensor shape with enough gap for this conv2d
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

    //Computes the new output size with updated input for updated output size may bigger than the smallest known size?
    uint32_t batch_fuse_w_update_output = 0;
    uint32_t batch_fuse_h_update_output = 0;
    int_pad[1] = int_pad[1] < 0 ? 0 : int_pad[1];
    int_pad[3] = int_pad[3] < 0 ? 0 : int_pad[3];
    batch_fuse_w_update_output =
        floor((float_t)(input_batch_fuse_w_update + int_pad[0] + int_pad[1] -
                        weight_shape[0]) /
              (float_t)(stride[0])) +
        1;
    batch_fuse_h_update_output =
        floor((float_t)(input_batch_fuse_h_update + int_pad[2] + int_pad[3] -
                        weight_shape[1]) /
              (float_t)(stride[1])) +
        1;

    vx::ShapeType output_batch_fuse_shape_update(4);
    output_batch_fuse_shape_update[w_axis] = batch_fuse_w_update_output;
    output_batch_fuse_shape_update[h_axis] = batch_fuse_h_update_output;
    output_batch_fuse_shape_update[c_axis] = output_shape[c_axis];
    output_batch_fuse_shape_update[batch_axis] = 1;

    //Update tensor shape map
    context_->UpdateGapInferShape(output_tensor,
                                  output_batch_fuse_shape_update);
    return need_backward;  // useless
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors,
      const std::shared_ptr<vx::Operation>& op,
      std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context) override {
    op_ = op;
    context_ = context;
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    auto output_shape = output_tensor->GetShape();
    auto weight_tensor = input_tensors[1];
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    auto weight_shape = weight_tensor->GetShape();

    // Original axis is [0, 1, 2, 3] -> [C, W, H, N]
    auto batch_src_axis = context_->GetBatchAxis();  // 3
    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    // after permute in layout inference, new axis is [1, 2, 0, 3] -> [W, H, C, N]
    // When we set batch_axis_ = 3, fuse_axes_ = [1, 2]
    // Here we can get mapped axis batch fuse want:
    auto perm_axis_map = context_->GetPermAxisMap(
        input_tensor);  //[1, 2, 0, 3] which represnet the index of original axis after permute
    auto fuse_axes = context_->GetPermFuseAxes(
        input_tensor);  //[0, 1], which represent the fuse's index of mapped axis in perm_axis_map
    auto batch_axis = context_->GetPermBatchAxis(
        input_tensor);  //3, which reprenstent the batch's index of mapped axis in perm_axis_map

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];

    //initialize the input tensor's gap infer shape
    context_->UpdateGapInferShape(input_batch_fuse_tensor,
                                  context_->GetGapInferShape(input_tensor));

    auto pad_type = OpBatchFuse::TranslatePadType(
        op_->impl()->node()->nn_param.conv2d.pad_type);
    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.conv2d.ksize[0],
        op_->impl()->node()->nn_param.conv2d.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.conv2d.stride[0],
        op_->impl()->node()->nn_param.conv2d.stride[1]};
    std::array<uint32_t, 2> dilation = {
        op_->impl()->node()->nn_param.conv2d.dilation[0],
        op_->impl()->node()->nn_param.conv2d.dilation[1]};
    std::array<uint32_t, 4> pad = {op_->impl()->node()->nn_param.conv2d.pad[0],
                                   op_->impl()->node()->nn_param.conv2d.pad[1],
                                   op_->impl()->node()->nn_param.conv2d.pad[2],
                                   op_->impl()->node()->nn_param.conv2d.pad[3]};

    std::array<int32_t, 4> int_pad = {0, 0, 0, 0};
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME || pad_type == vx::PadType::VALID) {
        int32_t p_w = stride[0] * output_shape[0] - input_shape[0] +
                      weight_shape[0] - stride[0];
        int32_t p_h = stride[1] * output_shape[1] - input_shape[1] +
                      weight_shape[1] - stride[1];

        int_pad[0] = p_w / 2;           //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_back
        int_pad[2] = p_h / 2;           //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_back

      } else {
        //TODO(HuanyuCai): AUTO how to pad?
        VSILOGW("AUTO pad is not supported yet");
      }
    }

    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;

    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    std::shared_ptr<vx::Tensor> conv2d_out_tensor;
    uint32_t out_w_batch = 0;
    uint32_t out_h_batch = 0;

    uint32_t batch = input_batch_fuse_shape[batch_axis];
    uint32_t batch_src = input_shape[batch_src_axis];

    if (batch != 1) {
      //input tensor has not been batch fused, fuse it

      //step one, if it need pad, pad it with input tensor's gap infer shape
      auto pad_tensor = InsertPad(input_batch_fuse_tensor, input_tensor,
                                  batch_axis, fuse_axes);

      //step two, batch fuse
      batch_fuse_tensor = InsertPermuteAndReshape(pad_tensor, input_tensor,
                                                  batch_axis, fuse_axes);

    } else {
      //input tensor has been fused
      batch_fuse_tensor = input_batch_fuse_tensor;
    }

    auto batch_fuse_shape = batch_fuse_tensor->GetShape();
    auto valid_ratio =
        (float)(input_shape[w_axis] * input_shape[h_axis] * batch_src) /
        (float)(batch_fuse_shape[w_axis] * batch_fuse_shape[h_axis]);
    context_->UpdateProportion(input_tensor, valid_ratio);

    //let floor() to dircard extra pixels rather negative pad
    int_pad[1] = int_pad[1] < 0 ? 0 : int_pad[1];
    int_pad[3] = int_pad[3] < 0 ? 0 : int_pad[3];
    out_w_batch =
        floor((float_t)(batch_fuse_shape[w_axis] - weight_shape[w_axis] +
                        int_pad[0] + int_pad[1]) /
              (float_t)(stride[0])) +
        1;
    out_h_batch =
        floor((float_t)(batch_fuse_shape[h_axis] - weight_shape[h_axis] +
                        int_pad[2] + int_pad[3]) /
              (float_t)(stride[1])) +
        1;

    tim::vx::ShapeType output_batch_fuse_shape(4, 0);
    output_batch_fuse_shape[w_axis] = out_w_batch;
    output_batch_fuse_shape[h_axis] = out_h_batch;
    uint32_t c_axis;
    for (uint32_t i(0); i < output_batch_fuse_shape.size(); i++) {
      if (i != batch_axis && i != w_axis && i != h_axis) {
        c_axis = i;
      }
    }
    output_batch_fuse_shape[c_axis] = weight_shape[batch_axis];
    output_batch_fuse_shape[batch_axis] = 1;

    auto output_spec = output_tensor->GetSpec();
    auto output_batch_fuse_spec = output_spec.SetShape(output_batch_fuse_shape);
    conv2d_out_tensor =
        context_->GetBatchFuseGraph()->CreateTensor(output_batch_fuse_spec);
    context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);

    //create new weight tensor and copy the constant value
    std::vector<uint8_t> tmp_weight(weight_tensor->GetSpec().GetByteSize());
    weight_tensor->CopyDataFromTensor(tmp_weight.data());
    auto batch_weight_tensor = context_->GetBatchFuseGraph()->CreateTensor(
        weight_tensor->GetSpec(), tmp_weight.data());
    context_->UpdateTensorMap(weight_tensor, batch_weight_tensor);

    if (input_tensors.size() > 2) {
      //create bias tensor and copy constant value
      auto bias_tensor = input_tensors[2];
      std::vector<uint8_t> tmp_bias(bias_tensor->GetSpec().GetByteSize());
      bias_tensor->CopyDataFromTensor(tmp_bias.data());
      auto batch_bias_tensor = context_->GetBatchFuseGraph()->CreateTensor(
          bias_tensor->GetSpec(), tmp_bias.data());
      context_->UpdateTensorMap(bias_tensor, batch_bias_tensor);
    }
    //insert mask to clean gap inside's value to be 0
    if (batch_fuse_shape[batch_axis] == 1 && batch_src != 1) {
      auto masked_input =
          InsertMask(batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
      context_->UpdateTensorMap(input_tensor, masked_input);
    }

    auto conv2d = context_->GetBatchFuseGraph()->CreateOperation<vx::ops::Conv2d>(
        out_channels, pad_type, ksize, stride, dilation, pad, multiplier,
        vx::DataLayout::WHCN, vx::DataLayout::WHIcOc);

    context_->UpdateTensorMap(output_tensor, conv2d_out_tensor);

    for (const auto& i_src : input_tensors) {
      (*conv2d).BindInput(context_->GetMapedTensor(i_src));
    }
    (*conv2d).BindOutput(conv2d_out_tensor);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(output_tensor);
  }
};
}  // namespace fuse
}  // namespace tim

#endif