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
#ifndef TIM_BATCH_FUSE_PADV2_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_PADV2_BATCH_FUSE_H_

#include "tim/vx/ops/pad.h"
// #include "tim/vx/ops/pad_v2.h"

#include "builtin_op_impl.h"
// #include "permute_vector.h"
#include "op_batch_fuse.h"

namespace tim {
namespace fuse {
class PadV2BatchFuse : public OpBatchFuse {
 public:
  PadV2BatchFuse(const std::shared_ptr<vx::Operation> op,
               std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    // auto init_pad = context_->GetInitPad(input_tensor);

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    auto batch_fuse_shape_input = context_->GetPadInferShape(input_tensor);
    uint32_t dim_num = op_->impl()->node()->nn_param.pad2.dim_num;

    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad2.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad2.back_size,
           sizeof(uint32_t) * dim_num);
    int32_t pad_value = op_->impl()->node()->nn_param.pad2.const_val;

    std::array<uint32_t, 4> pad = {front_size[0], front_size[1], back_size[0],
                                   back_size[1]};
    context_->UpdateInitPad(input_tensor, pad);

    auto old_pad_init = context_->GetForwardPad(input_tensor);
    for (int i = 0; i < pad.size(); i++) {
      if (pad[i] > old_pad_init[i]) {
        context_->UpdateForwardPad(input_tensor, pad);
        break;
      }
    }

    auto batch_fuse_shape_old = context_->GetPadInferShape(input_tensor);

    auto batch_fuse_w_old = batch_fuse_shape_old[0];
    auto batch_fuse_h_old = batch_fuse_shape_old[1];
    if (batch_fuse_shape_old[3] != 1) {
      //batch fuse
      batch_fuse_w_old *= batch_factor_w;
      batch_fuse_h_old *= batch_factor_h;
      batch_fuse_shape_old = {batch_fuse_w_old, batch_fuse_h_old,
                              batch_fuse_shape_old[2], 1};
      context_->UpdatePadInferShape(input_tensor, batch_fuse_shape_old);
    }

    auto batch_fuse_w_new = input_shape[0] * batch_factor_w +
                            (batch_factor_w - 1) * pad[1];  //pad[1] >= pad[0]
    auto batch_fuse_h_new = input_shape[1] * batch_factor_h +
                            (batch_factor_h - 1) * pad[3];  //pad[3] >= pad[2]
    vx::ShapeType batch_fuse_shape_new = {batch_fuse_w_new, batch_fuse_h_new,
                                          input_shape[2], 1};  //whcn, n = 1

    bool need_backward = false;
    uint32_t input_w, input_h;
    if (batch_fuse_w_new > batch_fuse_w_old ||
        batch_fuse_h_new > batch_fuse_h_old) {
      context_->UpdatePadInferShape(input_tensor, batch_fuse_shape_new);
      need_backward = true;  //need backward to update pad
      input_w = batch_fuse_w_new;
      input_h = batch_fuse_h_new;

    } else {
      input_w = batch_fuse_w_old;
      input_h = batch_fuse_h_old;
    }

    //update output tensor -> fused tensor shape with this temporary pad
    auto output_w = input_w + pad[0] + pad[1];
    auto output_h = input_h + pad[2] + pad[3];
    vx::ShapeType output_batch_fuse_shape = {output_w, output_h,
                                             output_shape[2], 1};  //whcn, n = 1
    context_->UpdatePadInferShape(output_tensor, output_batch_fuse_shape);
    context_->UpdateForwardPad(output_tensor, {0, 0, 0, 0});

    next_tensors.push_back(output_tensor);
    return need_backward;
  }
  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_tensor = input_tensors[0];
    // auto weight_tensor = input_tensors[1];
    // auto weight_shape = weight_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto init_pad = context_->GetInitPad(input_tensor);
    auto batch_fuse_shape_new = context_->GetPadInferShape(output_tensor);
    auto batch_fuse_shape_input = context_->GetPadInferShape(input_tensor);

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;
    uint32_t dim_num = op_->impl()->node()->nn_param.pad2.dim_num;
    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad2.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad2.back_size,
           sizeof(uint32_t) * dim_num);
    int32_t pad_value = op_->impl()->node()->nn_param.pad2.const_val;

    std::array<uint32_t, 4> pad = {front_size[0], back_size[0], front_size[1],
                                   back_size[1]};

    //Computes the size of the updated input using the output of the smallest known size
    uint32_t batch_fuse_w_update_input =
        batch_fuse_shape_new[0] - pad[0] - pad[1];
    uint32_t batch_fuse_h_update_input =
        batch_fuse_shape_new[1] - pad[2] - pad[3];

    //Computes the new pad size of updated input
    std::array<uint32_t, 4> new_pad = {0, 0, 0, 0};
    new_pad[1] = ceil((float_t)((batch_fuse_w_update_input -
                                 batch_factor_w * input_shape[0])) /
                      (float_t)(batch_factor_w - 1));
    new_pad[0] = new_pad[1];  //Greedy
    new_pad[3] = ceil((float_t)((batch_fuse_h_update_input -
                                 batch_factor_h * input_shape[0])) /
                      (float_t)(batch_factor_h - 1));
    new_pad[2] = new_pad[3];  //Greedy

    //Update pad map
    auto old_pad_forward = context_->GetForwardPad(input_tensor);
    // auto old_pad_backward = context_->GetBackwardPad(input_tensor);
    for (int i = 0; i < new_pad.size(); i++) {
      if (new_pad[i] > old_pad_forward[i]) {
        context_->UpdateBackwardPad(input_tensor, new_pad);
        context_->UpdateForwardPad(input_tensor, new_pad);
        break;
      }
    }

    bool need_backward = false;
    if ((batch_fuse_shape_input[0] < batch_fuse_w_update_input) ||
        (batch_fuse_shape_input[1] < batch_fuse_h_update_input)) {
      //continue backward
      vx::ShapeType input_batch_fuse_shape = {batch_fuse_w_update_input,
                                              batch_fuse_h_update_input,
                                              input_shape[2], 1};  //whcn, n = 1

      former_tensors.push_back(input_tensor);
      need_backward = true;
      context_->UpdatePadInferShape(input_tensor, input_batch_fuse_shape);
    }
    return need_backward;  // useless
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    std::shared_ptr<vx::Tensor> slice_and_concat_out;
    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    // std::shared_ptr<vx::Tensor> fc_out_tensor;

    //input_tesnor
    uint32_t batch = context_->GetMapedTensor(input_tensor)->GetShape()[3];
    uint32_t batch_src = input_tensor->GetShape()[3];
    // if (batch == 1 && batch_src != 1) {
    //   //insert slice and concat
    //   slice_and_concat_out =
    //       InsertSliceAndConcat(input_batch_fuse_tensor, false, input_tensor);
    // } else {
    //   slice_and_concat_out = input_batch_fuse_tensor;
    // }

    if (batch != 1) {
      auto pad_tensor = InsertPad(input_batch_fuse_tensor, false, input_tensor);
      batch_fuse_tensor =
          InsertPermuteAndReshape(pad_tensor, false, input_tensor);
    } else {
      batch_fuse_tensor = input_batch_fuse_tensor;
    }
    
    // slice_and_concat_out = batch_fuse_tensor;
    // auto slice_and_concat_shape = slice_and_concat_out->GetShape();
    // context_->UpdateTensorMap(input_tensor, slice_and_concat_out);
    // context_->UpdateTensorBatchFuseMap(slice_and_concat_out, input_tensor);
    auto batch_fuse_tensor_shape = batch_fuse_tensor->GetShape();
    context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    context_->UpdateTensorBatchFuseMap(batch_fuse_tensor, input_tensor);

    uint32_t dim_num = op_->impl()->node()->nn_param.pad2.dim_num;
    std::vector<uint32_t> front_size(dim_num, 0);
    std::vector<uint32_t> back_size(dim_num, 0);
    memcpy(front_size.data(), op_->impl()->node()->nn_param.pad2.front_size,
           sizeof(uint32_t) * dim_num);
    memcpy(back_size.data(), op_->impl()->node()->nn_param.pad2.back_size,
           sizeof(uint32_t) * dim_num);
    int32_t pad_value = op_->impl()->node()->nn_param.pad2.const_val;

    auto pad_op = context_->batch_fuse_graph_->CreateOperation<vx::ops::PadV2>(
        front_size, back_size, pad_value, tim::vx::ops::PadV2::PAD_MODE_CONSTANT);

    // auto out_de_batch_fuse = CreateOutputsTensor();
    // auto out_de_batch_shape = out_de_batch_fuse[0]->GetShape();
    auto out_w_batch = batch_fuse_tensor_shape[0] + front_size[1] + back_size[1];
    auto out_h_batch = batch_fuse_tensor_shape[1] + front_size[2] + back_size[2];
     tim::vx::ShapeType pad_output_shape(
        {out_w_batch, out_h_batch, batch_fuse_tensor_shape[2], 1});  //whcn
    // tim::vx::TensorSpec pad_output_spec(input_tensor->GetDataType(),
    //                                      pad_output_shape,
    //                                      tim::vx::TensorAttribute::TRANSIENT);
    auto output_spec = output_tensor->GetSpec();
    auto pad_output_spec = output_spec.SetShape(pad_output_shape);
    auto pad_out_tensor =
        context_->batch_fuse_graph_->CreateTensor(pad_output_spec);

    context_->UpdateTensorMap(op_->impl()->OutputsTensor()[0], pad_out_tensor);
    context_->UpdateTensorBatchFuseMap(pad_out_tensor, op_->impl()->OutputsTensor()[0]);
    for (auto in : op_->impl()->InputsTensor()) {
      (*pad_op).BindInput(context_->GetMapedTensor(in));
    }

    (*pad_op).BindOutput(pad_out_tensor);
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace fuse
}  // namespace tim

#endif