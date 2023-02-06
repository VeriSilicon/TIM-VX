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
// #include "permute_vector.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/pool2d.h"

namespace tim {
namespace fuse {
class Pool2dBatchFuse : public OpBatchFuse {
 public:
  Pool2dBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensor = input_tensors[0];
    // auto weight_tensor = input_tensors[1];
    auto input_shape = input_tensor->GetShape();
    auto output_shape = output_tensor->GetShape();  //whcn
    // auto weight_shape = weight_tensor->GetShape();

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.pool.ksize[0],
        op_->impl()->node()->nn_param.pool.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.pool.stride[0],
        op_->impl()->node()->nn_param.pool.stride[1]};
    // auto pool_type = TranslatePoolType(op_->impl()->node()->nn_param.pool.type);
    // auto round_type =
    //     TranslateRoundType(op_->impl()->node()->nn_param.pool.round_type);
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
        //cal pad and map init pad
        int32_t p_w =
            stride[0] * output_shape[0] - input_shape[0] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[1] - input_shape[1] + ksize[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        int_pad[0] = p_w / 2;       //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_end
        int_pad[2] = p_h / 2;       //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }
    if (int_pad[0] > 0 || int_pad[1] > 0 || int_pad[2] > 0 || int_pad[3] > 0){
      // do not batch fuse and pad inside
      // context_->UpdateInitPad(input_tensor, pad);
      // context_->UpdateForwardPad(input_tensor, {0, 0, 0, 0});
      context_->UpdatePadInferShape(input_tensor, input_shape);
      context_->UpdatePadInferShape(output_tensor, output_shape);
      // context_->UpdateForwardPad(output_tensor, {0, 0, 0, 0});

      context_->UpdateForwardGap(input_tensor, {0, 0});
      context_->UpdateForwardGap(output_tensor, {0, 0});
      next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
      return false;
    }
    //update init pad
    // context_->UpdateInitPad(input_tensor, pad);
    // auto old_pad_init = context_->GetForwardPad(input_tensor);
    // for (uint i = 0; i < pad.size(); i++) {
    //   if (pad[i] > old_pad_init[i]) {
    //     context_->UpdateForwardPad(input_tensor, pad);
    //     break;
    //   }
    // }

    //update input tensor -> fused tensor shape with this temporary pad as the smallest know size
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
    
    std::array<uint32_t, 2> gap = {0, 0};
    auto m_w = ceil((float_t)(ksize[0] - int_pad[0] - int_pad[1]) / (float_t)stride[0]);
    auto m_h = ceil((float_t)(ksize[1] - int_pad[2] - int_pad[3]) / (float_t)stride[1]);
    gap[0] = (m_w + 2) * stride[0] + int_pad[0] + int_pad[1] - ksize[0];
    gap[1] = (m_h + 2) * stride[1] + int_pad[2] + int_pad[3] - ksize[1];
    context_->UpdateForwardGap(input_tensor, gap);
    auto batch_fuse_w_new =
        input_shape[0] * batch_factor_w + (batch_factor_w - 1) * gap[0];
    auto batch_fuse_h_new =
        input_shape[1] * batch_factor_h + (batch_factor_h - 1) * gap[1];

    
    // auto batch_fuse_w_new = input_shape[0] * batch_factor_w +
    //                         (batch_factor_w - 1) * pad[1];  //pad[1] >= pad[0]
    // auto batch_fuse_h_new = input_shape[1] * batch_factor_h +
    //                         (batch_factor_h - 1) * pad[3];  //pad[3] >= pad[2]
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
      //old(may have been updated) tensor size > new(the smallest known) size, use old size as input
      input_w = batch_fuse_w_old;
      input_h = batch_fuse_h_old;
    }

    //cal the batch fused output shape of this conv2d
    uint32_t batch_fuse_w_update_output = 0;
    uint32_t batch_fuse_h_update_output = 0;
    batch_fuse_w_update_output = ceil(
        (float_t)(input_w - ksize[0] + int_pad[0] + int_pad[1]) / (float_t)(stride[0]) +
        1);
    batch_fuse_h_update_output = ceil(
        (float_t)(input_h - ksize[1] + int_pad[2] + int_pad[3]) / (float_t)(stride[1]) +
        1);

    vx::ShapeType batch_fuse_shape_update = {input_w, input_h, input_shape[2],
                                             1};  //whcn, n = 1
    context_->UpdatePadInferShape(input_tensor, batch_fuse_shape_update);

    //update output tensor -> fused tensor shape with this temporary pad
    vx::ShapeType output_batch_fuse_shape = {batch_fuse_w_update_output,
                                             batch_fuse_h_update_output,
                                             output_shape[2], 1};  //whcn, n = 1
    context_->UpdatePadInferShape(output_tensor, output_batch_fuse_shape);
    // context_->UpdateForwardPad(output_tensor, {0, 0, 0, 0});
    auto out_gap_w = (batch_fuse_w_update_output - output_shape[0] * batch_factor_w) / (batch_factor_w - 1);
    auto out_gap_h = (batch_fuse_w_update_output - output_shape[0] * batch_factor_w) / (batch_factor_w - 1);
    std::array<uint32_t, 2> output_gap = {out_gap_w, out_gap_h};
    context_->UpdateForwardGap(output_tensor, output_gap);

    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
    // former_tensors.push_back(op_->impl()->OutputsTensor()[0]);
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
    // auto init_pad = context_->GetInitPad(input_tensor);
    auto batch_fuse_shape_new = context_->GetPadInferShape(output_tensor);
    auto batch_fuse_shape_input = context_->GetPadInferShape(input_tensor);

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    std::array<uint32_t, 2> ksize = {
        op_->impl()->node()->nn_param.pool.ksize[0],
        op_->impl()->node()->nn_param.pool.ksize[1]};
    std::array<uint32_t, 2> stride = {
        op_->impl()->node()->nn_param.pool.stride[0],
        op_->impl()->node()->nn_param.pool.stride[1]};
    // auto pool_type = TranslatePoolType(op_->impl()->node()->nn_param.pool.type);
    // auto round_type =
    //     TranslateRoundType(op_->impl()->node()->nn_param.pool.round_type);
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
        //cal pad and map init pad
        int32_t p_w =
            stride[0] * output_shape[0] - input_shape[0] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[1] - input_shape[1] + ksize[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        int_pad[0] = p_w / 2;       //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_end
        int_pad[2] = p_h / 2;       //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }

    if (int_pad[0] > 0 || int_pad[1] > 0 || int_pad[2] > 0 || int_pad[3] > 0){
      context_->UpdateForwardGap(output_tensor, context_->GetForwardGap(output_tensor));
      context_->UpdatePadInferShape(output_tensor, context_->GetPadInferShape(output_tensor));
      return false;
    }
    
    std::array<uint32_t, 2> gap_input = {0, 0};
    std::array<uint32_t, 2> gap_output = context_->GetForwardGap(output_tensor);
    gap_input[0] = stride[0] * (gap_output[0] + 1) + int_pad[0] + int_pad[1];
    gap_input[1] = stride[1] * (gap_output[1] + 1) + int_pad[2] + int_pad[3];
    context_->UpdateForwardGap(input_tensor, gap_input);

    uint32_t batch_fuse_w_update_input =
        input_shape[0] * batch_factor_w + (batch_factor_w - 1) * gap_input[0];

    uint32_t batch_fuse_h_update_input =
        input_shape[1] * batch_factor_h + (batch_factor_h - 1) * gap_input[1];
    
    //Computes the size of the updated input using the output of the smallest known size
    // uint32_t batch_fuse_w_update_input = stride[0] * batch_fuse_shape_new[0] -
    //                                      pad[0] - pad[1] + ksize[0] - stride[0];
    // uint32_t batch_fuse_h_update_input = stride[1] * batch_fuse_shape_new[1] -
    //                                      pad[2] - pad[3] + ksize[1] - stride[1];

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
    // auto old_pad_forward = context_->GetForwardPad(input_tensor);
    // // auto old_pad_backward = context_->GetBackwardPad(input_tensor);
    // for (uint i = 0; i < new_pad.size(); i++) {
    //   if (new_pad[i] > old_pad_forward[i]) {
    //     context_->UpdateBackwardPad(input_tensor, new_pad);
    //     context_->UpdateForwardPad(input_tensor, new_pad);
    //     break;
    //   }
    // }

    // uint32_t batch_fuse_w_update_input =
    //     input_shape[0] * batch_factor_w + (batch_factor_w - 1) * new_pad[1];
    // uint32_t batch_fuse_h_update_input =
    //     input_shape[1] * batch_factor_h + (batch_factor_h - 1) * new_pad[3];

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

    //Computes the new output size with updated input for updated output size may bigger than the smallest known size
    uint32_t batch_fuse_w_update_output = 0;
    uint32_t batch_fuse_h_update_output = 0;

    batch_fuse_w_update_output =
        ceil((float_t)(batch_fuse_w_update_input + int_pad[0] + int_pad[1] - ksize[0]) /
             (float_t)(stride[0])) +
        1;
    batch_fuse_h_update_output =
        ceil((float_t)(batch_fuse_h_update_input + int_pad[2] + int_pad[3] - ksize[1]) /
             (float_t)(stride[1])) +
        1;

    vx::ShapeType output_batch_fuse_shape = {batch_fuse_w_update_output,
                                             batch_fuse_h_update_output,
                                             output_shape[2], 1};  //whcn, n = 1

    //Update tensor shape map
    context_->UpdatePadInferShape(output_tensor, output_batch_fuse_shape);
    return need_backward;  // useless
  }
  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    // vx::DataLayout layout = op_->impl()->layout_;
    auto input_tensor = op_->impl()->InputsTensor()[0];
    auto input_tensor_shape = input_tensor->GetSpec();
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    auto batch_src = input_tensor->GetShape()[3];
    auto batch = input_batch_fuse_tensor->GetShape()[3];
    auto channel = input_batch_fuse_tensor->GetShape()[2];

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
        //cal pad and map init pad
        int32_t p_w =
            stride[0] * output_shape[0] - input_shape[0] + ksize[0] - stride[0];
        int32_t p_h =
            stride[1] * output_shape[1] - input_shape[1] + ksize[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        int_pad[0] = p_w / 2;       //p_w_front
        int_pad[1] = p_w - int_pad[0];  //p_w_end
        int_pad[2] = p_h / 2;       //p_h_front
        int_pad[3] = p_h - int_pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }

    // auto sqrt_batch = sqrt(batch_src);
    // uint32_t batch_factor_w = ClosestFactors(batch).first;
    // uint32_t batch_factor_h = ClosestFactors(batch).second;
    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    std::shared_ptr<vx::Tensor> pool2d_out_tensor;
    uint32_t out_w_batch = 0;
    uint32_t out_h_batch = 0;

    if (batch != 1) {
      if (int_pad[0] <= 0 && int_pad[1] <= 0 && int_pad[2] <= 0 && int_pad[3] <= 0)
      // pool wont be effected by pad value
      {
        auto pad_tensor =
            InsertPad(input_batch_fuse_tensor, input_tensor);
        batch_fuse_tensor =
            InsertPermuteAndReshape(pad_tensor, input_tensor);
      } else {
        // do not batch fuse
        batch_fuse_tensor = input_tensor;
      }
    } else {
      //fused before
      if (int_pad[0] <= 0 && int_pad[1] <= 0 && int_pad[2] <= 0 && int_pad[3] <= 0) {
        batch_fuse_tensor = input_batch_fuse_tensor;
      } else {
        // do not batch fuse
        batch_fuse_tensor = InsertSliceAndConcat(input_batch_fuse_tensor, input_tensor);
      }
    }
    auto batch_fuse_shape = batch_fuse_tensor->GetShape();

    out_w_batch =
        ceil((float_t)(batch_fuse_shape[0] - ksize[0] + int_pad[0] + int_pad[1]) /
             (float_t)(stride[0])) +
        1;
    out_h_batch =
        ceil((float_t)(batch_fuse_shape[1] - ksize[1] + int_pad[2] + int_pad[3]) /
             (float_t)(stride[1])) +
        1;

    tim::vx::ShapeType pool_output_shape(
        {out_w_batch, out_h_batch, channel, batch_fuse_shape[3]});  //whcn
    
    auto output_spec = output_tensor->GetSpec();
    auto pool_output_spec = output_spec.SetShape(pool_output_shape);
    pool2d_out_tensor =
        context_->batch_fuse_graph_->CreateTensor(pool_output_spec);
    context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    context_->UpdateTensorBatchFuseMap(batch_fuse_tensor, input_tensor);

    context_->UpdateTensorMap(output_tensor, pool2d_out_tensor);
    context_->UpdateTensorBatchFuseMap(pool2d_out_tensor, output_tensor);

    //inser mask
    if (batch_fuse_tensor->GetShape()[3] == 1 && batch_src != 1) {
      auto masked_input = InsertMask(batch_fuse_tensor, input_tensor);
      context_->UpdateTensorMap(input_tensor, masked_input);
      // context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    }

    auto pool2d = context_->batch_fuse_graph_->CreateOperation<vx::ops::Pool2d>(
        pool_type, pad_type, ksize, stride, round_type, vx::DataLayout::WHCN);

    (*pool2d).BindInput(context_->GetMapedTensor(input_tensor));
    (*pool2d).BindOutput(pool2d_out_tensor);

    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};

}  // namespace fuse
}  // namespace tim

#endif