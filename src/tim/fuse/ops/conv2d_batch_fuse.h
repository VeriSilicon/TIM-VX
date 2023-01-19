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
#ifndef TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_CONV2D_BATCH_FUSE_H_

#include "tim/vx/ops/conv2d.h"

#include "builtin_op_impl.h"
// #include "permute_vector.h"
#include "op_batch_fuse.h"

namespace tim {
namespace fuse {
class Conv2dBatchFuse : public OpBatchFuse {
 public:
  Conv2dBatchFuse(const std::shared_ptr<vx::Operation> op,
                  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : OpBatchFuse(op, context) {}

  bool PadForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
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
    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME) {
        //cal pad and map init pad
        auto p_w = stride[0] * output_shape[0] - input_shape[0] +
                   weight_shape[0] - stride[0];
        auto p_h = stride[1] * output_shape[1] - input_shape[1] +
                   weight_shape[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        pad[0] = p_w / 2;       //p_w_front
        pad[1] = p_w - pad[0];  //p_w_end
        pad[2] = p_h / 2;       //p_h_front
        pad[3] = p_h - pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }
    //update init pad
    context_->UpdateInitPad(input_tensor, pad);
    auto old_pad_init = context_->GetForwardPad(input_tensor);
    for (int i = 0; i < pad.size(); i++) {
      if (pad[i] > old_pad_init[i]) {
        context_->UpdateForwardPad(input_tensor, pad);
        break;
      }
    }

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
      //old(may have been updated) tensor size > new(the smallest known) size, use old size as input
      input_w = batch_fuse_w_old;
      input_h = batch_fuse_h_old;
    }

    //cal the batch fused output shape of this conv2d
    uint32_t batch_fuse_w_update_output = 0;
    uint32_t batch_fuse_h_update_output = 0;
    batch_fuse_w_update_output =
        floor((float_t)(input_w - weight_shape[0] + pad[0] + pad[1]) /
                  (float_t)(stride[0]) +
              1);
    batch_fuse_h_update_output =
        floor((float_t)(input_h - weight_shape[1] + pad[2] + pad[3]) /
                  (float_t)(stride[1]) +
              1);

    //update output tensor -> fused tensor shape with this temporary pad
    vx::ShapeType output_batch_fuse_shape = {batch_fuse_w_update_output,
                                             batch_fuse_h_update_output,
                                             output_shape[2], 1};  //whcn, n = 1
    context_->UpdatePadInferShape(output_tensor, output_batch_fuse_shape);
    context_->UpdateForwardPad(output_tensor, {0, 0, 0, 0});

    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
    // former_tensors.push_back(op_->impl()->OutputsTensor()[0]);
    return need_backward;
  }

  bool PadBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto input_tensor = input_tensors[0];
    auto weight_tensor = input_tensors[1];
    auto weight_shape = weight_tensor->GetShape();
    auto input_shape = input_tensor->GetShape();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto output_shape = output_tensor->GetShape();
    auto init_pad = context_->GetInitPad(input_tensor);
    auto batch_fuse_shape_new = context_->GetPadInferShape(output_tensor);
    auto batch_fuse_shape_input = context_->GetPadInferShape(input_tensor);

    uint32_t batch = input_tensor->GetShape()[3];
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

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
    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME) {
        //cal pad and map init pad
        auto p_w = stride[0] * output_shape[0] - input_shape[0] +
                   weight_shape[0] - stride[0];
        auto p_h = stride[1] * output_shape[1] - input_shape[1] +
                   weight_shape[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        pad[0] = p_w / 2;       //p_w_front
        pad[1] = p_w - pad[0];  //p_w_end
        pad[2] = p_h / 2;       //p_h_front
        pad[3] = p_h - pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }

    //Computes the size of the updated input using the output of the smallest known size
    uint32_t batch_fuse_w_update_input = stride[0] * batch_fuse_shape_new[0] -
                                         pad[0] - pad[1] + weight_shape[0] -
                                         stride[0];
    uint32_t batch_fuse_h_update_input = stride[1] * batch_fuse_shape_new[1] -
                                         pad[2] - pad[3] + weight_shape[1] -
                                         stride[1];

    //Computes the new pad size of updated input
    std::array<uint32_t, 4> new_pad = {0, 0, 0, 0};
    new_pad[1] = floor((float_t)((batch_fuse_w_update_input -
                                 batch_factor_w * input_shape[0])) /
                      (float_t)(batch_factor_w - 1));
    new_pad[0] = new_pad[1];  //Greedy
    new_pad[3] = floor((float_t)((batch_fuse_h_update_input -
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
        floor((float_t)(batch_fuse_w_update_input + pad[0] + pad[1] -
                        weight_shape[0]) /
              (float_t)(stride[0])) +
        1;
    batch_fuse_h_update_output =
        floor((float_t)(batch_fuse_h_update_input + pad[2] + pad[3] -
                        weight_shape[1]) /
              (float_t)(stride[1])) +
        1;
    vx::ShapeType output_batch_fuse_shape = {batch_fuse_w_update_output,
                                             batch_fuse_h_update_output,
                                             output_shape[2], 1};  //whcn, n = 1

    //Update tensor shape map
    context_->UpdatePadInferShape(output_tensor, output_batch_fuse_shape);
    return need_backward; // useless
  }

  void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) override {
    auto input_tensors = op_->impl()->InputsTensor();
    auto output_tensor = op_->impl()->OutputsTensor()[0];
    auto input_tensor = input_tensors[0];
    auto input_shape = input_tensor->GetShape();
    auto output_shape = output_tensor->GetShape();
    auto weight_tensor = input_tensors[1];
    auto input_batch_fuse_tensor = context_->GetMapedTensor(input_tensor);
    auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();
    auto weight_shape = weight_tensor->GetShape();
    auto bias_tensor = input_tensors[2];
    auto bias_shape = bias_tensor->GetShape();

    context_->UpdatePadInferShape(input_batch_fuse_tensor, context_->GetPadInferShape(input_tensor));

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
    if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
      if (pad_type == vx::PadType::SAME) {
        //cal pad and map init pad
        auto p_w = stride[0] * output_shape[0] - input_shape[0] +
                   weight_shape[0] - stride[0];
        auto p_h = stride[1] * output_shape[1] - input_shape[1] +
                   weight_shape[1] - stride[1];

        //pad[1] >= pad[0], pad[3] >= pad[2]
        pad[0] = p_w / 2;       //p_w_front
        pad[1] = p_w - pad[0];  //p_w_end
        pad[2] = p_h / 2;       //p_h_front
        pad[3] = p_h - pad[2];  //p_h_end

      } else {
        //AUTO how to pad?
      }
    }

    int32_t multiplier = op_->impl()->node()->nn_param.conv2d.multiplier;
    int32_t out_channels = op_->impl()->node()->nn_param.conv2d.weights;

    std::shared_ptr<vx::Tensor> batch_fuse_tensor;
    std::shared_ptr<vx::Tensor> conv2d_out_tensor;
    uint32_t out_w_batch = 0;
    uint32_t out_h_batch = 0;

    uint32_t batch = input_batch_fuse_tensor->GetShape()[3];
    uint32_t batch_src = input_tensor->GetShape()[3];
    // auto sqrt_batch = sqrt(batch);
    uint32_t batch_factor_w = ClosestFactors(batch).first;
    uint32_t batch_factor_h = ClosestFactors(batch).second;

    if (batch != 1) {
      // batch fuse
      auto pad_tensor = InsertPad(input_batch_fuse_tensor, false, input_tensor);
      batch_fuse_tensor = InsertPermuteAndReshape(pad_tensor, false, input_tensor);

    } else {
      batch_fuse_tensor = input_batch_fuse_tensor;
    }

    auto batch_fuse_shape = batch_fuse_tensor->GetShape();
    out_w_batch = floor((float_t)(batch_fuse_shape[0] - weight_shape[0] +
                                  pad[0] + pad[1]) /
                        (float_t)(stride[0])) +
                  1;
    out_h_batch = floor((float_t)(batch_fuse_shape[1] - weight_shape[1] +
                                  pad[2] + pad[3]) /
                        (float_t)(stride[1])) +
                  1;

    tim::vx::ShapeType conv_output_shape(
        {out_w_batch, out_h_batch, weight_shape[3], 1});  //whcn
    // tim::vx::TensorSpec conv_output_spec(input_tensor->GetDataType(),
    //                                      conv_output_shape,
    //                                      tim::vx::TensorAttribute::TRANSIENT);
    auto output_spec = output_tensor->GetSpec();
    auto conv_output_spec = output_spec.SetShape(conv_output_shape);
    conv2d_out_tensor =
        context_->batch_fuse_graph_->CreateTensor(conv_output_spec);

    context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    context_->UpdateTensorBatchFuseMap(batch_fuse_tensor, input_tensor);
    
    std::vector<uint8_t> tmp_weight(weight_tensor->GetSpec().GetByteSize());
    weight_tensor->CopyDataFromTensor(tmp_weight.data());
    auto batch_weight_tensor = context_->batch_fuse_graph_->CreateTensor(weight_tensor->GetSpec(), tmp_weight.data());
    context_->UpdateTensorMap(weight_tensor, batch_weight_tensor);
    context_->UpdateTensorBatchFuseMap(batch_weight_tensor, weight_tensor);

    if (input_tensors.size() > 2) {
      auto bias_tensor = input_tensors[2];
      std::vector<uint8_t> tmp_bias(bias_tensor->GetSpec().GetByteSize());
      bias_tensor->CopyDataFromTensor(tmp_bias.data());
      auto batch_bias_tensor = context_->batch_fuse_graph_->CreateTensor(bias_tensor->GetSpec(), tmp_bias.data());
      context_->UpdateTensorMap(bias_tensor, batch_bias_tensor);
      context_->UpdateTensorBatchFuseMap(batch_bias_tensor, bias_tensor);
    }
    //inser mask
    if (batch_fuse_shape[3] == 1 && batch_src != 1){
      auto masked_input = InsertMask(batch_fuse_tensor, false, input_tensor);
      context_->UpdateTensorMap(input_tensor, masked_input);
    // context_->UpdateTensorMap(input_tensor, batch_fuse_tensor);
    }

    auto conv2d = context_->batch_fuse_graph_->CreateOperation<vx::ops::Conv2d>(
        out_channels, vx::PadType::AUTO, ksize, stride, dilation, pad, multiplier,
        vx::DataLayout::WHCN, vx::DataLayout::WHIcOc);

    context_->UpdateTensorMap(output_tensor, conv2d_out_tensor);
    context_->UpdateTensorBatchFuseMap(conv2d_out_tensor, output_tensor);

    for (const auto& i_src : input_tensors) {
      (*conv2d).BindInput(context_->GetMapedTensor(i_src));
    }
    (*conv2d).BindOutput(conv2d_out_tensor);
    // Add out tensor of src_graph into next_tensor
    next_tensors.push_back(op_->impl()->OutputsTensor()[0]);
  }
};
}  // namespace fuse
}  // namespace tim

#endif