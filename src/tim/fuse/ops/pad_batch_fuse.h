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
#ifndef TIM_BATCH_FUSE_PAD_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_PAD_BATCH_FUSE_H_

#include "tim/vx/ops/pad.h"
#include "builtin_op_impl.h"
#include "op_batch_fuse.h"

namespace tim {
namespace fuse {
class PadBatchFuse : public OpBatchFuse {
 public:
  PadBatchFuse(const std::shared_ptr<vx::Operation> op,
               std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();

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

    uint32_t dim_num = op_->impl()->node()->nn_param.pad.dim_num;
    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad.back_size,
           sizeof(uint32_t) * dim_num);

    std::array<uint32_t, 4> pad = {front_size[w_axis], back_size[w_axis], front_size[h_axis],
                                   back_size[h_axis]};

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

    //We do not reuse gap to both pad left and right, which means gap[0] >= pad[1] (pad[1] >= pad[0]), instead,
    //we reserve enough gap to pad left and right, that means gap[0] >= pad[0] + pad[1]
    auto input_batch_fuse_w_new = input_shape[w_axis] * batch_factor_w +
                                  (batch_factor_w - 1) * (pad[1] + pad[0]);
    auto input_batch_fuse_h_new = input_shape[h_axis] * batch_factor_h +
                                  (batch_factor_h - 1) * (pad[3] + pad[2]);
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
      context_->UpdateForwardGap(input_tensor, {pad[1] + pad[0], pad[3] + pad[2]});
      need_backward = true;  //need backward to update gap
      input_w = input_batch_fuse_w_new;
      input_h = input_batch_fuse_h_new;

    } else {
      input_w = input_batch_fuse_w_old;
      input_h = input_batch_fuse_h_old;
    }

    //Update output tensor -> fused tensor shape
    auto output_batch_fuse_w = input_w + pad[0] + pad[1];
    auto output_batch_fuse_h = input_h + pad[2] + pad[3];


    vx::ShapeType output_batch_fuse_shape(4);
    output_batch_fuse_shape[w_axis] = output_batch_fuse_w;
    output_batch_fuse_shape[h_axis] = output_batch_fuse_h;
    output_batch_fuse_shape[c_axis] = output_shape[c_axis];
    output_batch_fuse_shape[batch_axis] = 1;
    
    context_->UpdateGapInferShape(output_tensor, output_batch_fuse_shape);

    //Update output's gap
    std::array<uint32_t, 2> gap_output = {0, 0};
    gap_output[0] = ceil((float_t)((output_batch_fuse_w - batch_factor_w * output_shape[w_axis])) /
                      (float_t)(batch_factor_w - 1));
 
    gap_output[1] = ceil((float_t)((output_batch_fuse_h - batch_factor_h * output_shape[h_axis])) /
                      (float_t)(batch_factor_h - 1));
    
    context_->UpdateForwardGap(output_tensor, gap_output);
    next_tensors.push_back(output_tensor);
    return need_backward;
  }
  bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();

    auto output_batch_fuse_shape = context_->GetGapInferShape(output_tensor);
    auto input_batch_fuse_shape = context_->GetGapInferShape(input_tensor);

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
    uint32_t dim_num = op_->impl()->node()->nn_param.pad.dim_num;
    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad.back_size,
           sizeof(uint32_t) * dim_num);

    std::array<uint32_t, 4> pad = {front_size[w_axis], back_size[w_axis], front_size[h_axis],
                                   back_size[h_axis]};

    //Computes the size of the updated input using the output of the smallest known size
    uint32_t input_batch_fuse_w_update =
        output_batch_fuse_shape[w_axis] - pad[0] - pad[1];
    uint32_t input_batch_fuse_h_update =
        output_batch_fuse_shape[h_axis] - pad[2] - pad[3];

    //Computes the new pad size of updated input
    std::array<uint32_t, 2> gap_input = {0, 0};
    gap_input[0] = ceil((float_t)((input_batch_fuse_w_update -
                                 batch_factor_w * input_shape[w_axis])) /
                      (float_t)(batch_factor_w - 1));
    gap_input[1] = ceil((float_t)((input_batch_fuse_h_update -
                                 batch_factor_w * input_shape[h_axis])) /
                      (float_t)(batch_factor_h - 1));

    context_->UpdateForwardGap(input_tensor, gap_input);

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
      context_->UpdateGapInferShape(input_tensor, input_batch_fuse_shape_update);
    }
    return need_backward;  // useless
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();

    auto fuse_src_axes = context_->GetFuseAxes();    // [1, 2]

    auto perm_axis_map = context_->GetPermAxisMap(input_tensor);
    auto fuse_axes = context_->GetPermFuseAxes(input_tensor);
    auto batch_axis = context_->GetPermBatchAxis(input_tensor);
    auto c_axis = context_->GetPermChannelAxis(input_tensor);

    auto w_axis = fuse_axes[0];
    auto h_axis = fuse_axes[1];
   
    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    uint32_t batch = input_batch_fuse_shape[batch_axis];

    if (batch != 1) {
      //input tensor has not been batch fused, fuse it

      //step one, if it need pad, pad it with input tensor's gap infer shape
      auto pad_tensor = InsertPad(input_batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);

      //step two, batch fuse
      batch_fuse_tensor = InsertPermuteAndReshape(pad_tensor, input_tensor, batch_axis, fuse_axes);
    } else {
      batch_fuse_tensor = input_batch_fuse_tensor;
    }

    //insert mask to clean gap inside's value to be 0
    auto masked_input = InsertMask(batch_fuse_tensor, input_tensor, batch_axis, fuse_axes);
    context_->UpdateTensorMap(input_tensor, masked_input);

    auto batch_fuse_tensor_shape = masked_input->GetShape();

    auto valid_prop =
        (float)(input_shape[w_axis] * input_shape[h_axis] * input_shape[batch_axis]) /
        (float)(batch_fuse_tensor_shape[w_axis] * batch_fuse_tensor_shape[h_axis]);
    context_->UpdateProportion(input_tensor, valid_prop);

    uint32_t dim_num = op_->impl()->node()->nn_param.pad.dim_num;
    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad.back_size,
           sizeof(uint32_t) * dim_num);
    int32_t pad_value = op_->impl()->node()->nn_param.pad.const_val;

    auto pad_op = context_->batch_fuse_graph_->CreateOperation<vx::ops::Pad>(
        front_size, back_size, pad_value);

    //Pad batch fused tensor outside and compute its new shape
    auto out_w_batch =
        batch_fuse_tensor_shape[0] + front_size[w_axis] + back_size[w_axis];
    auto out_h_batch =
        batch_fuse_tensor_shape[1] + front_size[h_axis] + back_size[h_axis];
    tim::vx::ShapeType pad_output_shape(4);
    pad_output_shape[w_axis] = out_w_batch;
    pad_output_shape[h_axis] = out_h_batch;
    pad_output_shape[c_axis] = batch_fuse_tensor_shape[c_axis];
    pad_output_shape[batch_axis] = 1;
    
    auto output_spec = output_tensor->GetSpec();
    auto pad_output_spec = output_spec.SetShape(pad_output_shape);
    auto pad_out_tensor =
        context_->batch_fuse_graph_->CreateTensor(pad_output_spec);

    context_->UpdateTensorMap(output_tensor, pad_out_tensor);
    for (auto i_src : op_->impl()->InputsTensor()) {
      (*pad_op).BindInput(context_->GetMapedTensor(i_src));
    }

    (*pad_op).BindOutput(pad_out_tensor);
    next_tensors.push_back(output_tensor);
  }
};
}  // namespace fuse
}  // namespace tim

#endif