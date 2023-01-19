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

#include "op_batch_fuse.h"
// #include "permute_vector.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/pad.h"
// #include "tim/vx/ops/pad_v2.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/elementwise.h"
#include "type_utils.h"

#include <algorithm>
#include <vector>

namespace tim {
namespace fuse {

std::pair<uint32_t, uint32_t> OpBatchFuse::ClosestFactors(uint32_t n) {
  uint32_t int_sqrt = sqrt(n);
  uint32_t i = 0;
  if (n % int_sqrt == 0)
    return std::make_pair(int_sqrt, n / int_sqrt);
  else {
    uint32_t flag;
    uint32_t low = pow(int_sqrt, 2);
    uint32_t high = pow(int_sqrt + 1, 2);
    if ((n - low) >= (high - n))
      flag = int_sqrt + 1;
    else
      flag = int_sqrt;
    for (i = flag - 1; i > 0; i--) {
      if (n % i == 0) break;
    }
    return std::make_pair(i, n / i);
  }
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertPermuteAndReshape(
    std::shared_ptr<vx::Tensor> input, bool is_graph_output,
    std::shared_ptr<vx::Tensor> src_input) {
  auto in_spec = input->GetSpec();    // whcn
  auto in_shape = input->GetShape();  // whcn
  auto pad_infer_shape =
      context_->GetPadInferShape(src_input);  //inside paded tensor
  uint32_t batch_factor_w = ClosestFactors(in_shape[3]).first;
  uint32_t batch_factor_h = ClosestFactors(in_shape[3]).second;
  uint32_t in_channel = in_shape[2];
  // uint32_t out_channel = in_shape[3];
  uint32_t h = in_shape[1];
  uint32_t w = in_shape[0];
  std::vector<uint32_t> perm1 = {0, 1, 3, 4, 2, 5};
  std::vector<uint32_t> perm2 = {0, 2, 1, 3, 4, 5};
  // std::vector<uint32_t>& perm1 = perm_1;
  // std::vector<uint32_t>& perm2 = perm_2;

  vx::ShapeType reshape_1_shape(
      {w, h, in_channel, batch_factor_w, batch_factor_h, 1});  //whcbbn
  vx::ShapeType transpose_1_shape(
      {w, h, batch_factor_w, batch_factor_h, in_channel, 1});  //whbbcn
  vx::ShapeType transpose_2_shape(
      {w, batch_factor_w, h, batch_factor_h, in_channel, 1});  //wbhbcn
  vx::ShapeType reshape_2_shape(
      {w * batch_factor_w, h * batch_factor_h, in_channel, 1});  //whcn

  auto reshape_1_spec = in_spec.SetShape(reshape_1_shape);
  auto transpose_1_spec = in_spec.SetShape(transpose_1_shape);
  auto transpose_2_spec = in_spec.SetShape(transpose_2_shape);
  auto reshape_2_spec = in_spec.SetShape(reshape_2_shape);

  // vx::TensorSpec reshape_1_spec(in_spec.datatype_, reshape_1_shape,
  //                               tim::vx::TensorAttribute::TRANSIENT);
  // vx::TensorSpec transpose_1_spec(in_spec.datatype_, transpose_1_shape,
  //                                 tim::vx::TensorAttribute::TRANSIENT);
  // vx::TensorSpec transpose_2_spec(in_spec.datatype_, transpose_2_shape,
  //                                 tim::vx::TensorAttribute::TRANSIENT);
  // vx::TensorSpec reshape_2_spec(in_spec.datatype_, reshape_2_shape,
  //                               tim::vx::TensorAttribute::TRANSIENT);

  auto reshape_1_tensor =
      context_->batch_fuse_graph_->CreateTensor(reshape_1_spec);
  auto reshape_2_tensor =
      context_->batch_fuse_graph_->CreateTensor(reshape_2_spec);
  auto transpose_1_tensor =
      context_->batch_fuse_graph_->CreateTensor(transpose_1_spec);
  auto transpose_2_tensor =
      context_->batch_fuse_graph_->CreateTensor(transpose_2_spec);

  auto reshape_op_1 =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Reshape>(
          reshape_1_shape);
  auto reshape_op_2 =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Reshape>(
          reshape_2_shape);
  auto perm_op_1 =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Transpose>(perm1);
  auto perm_op_2 =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Transpose>(perm2);

  (*reshape_op_1).BindInput(input).BindOutput(reshape_1_tensor);
  (*perm_op_1).BindInput(reshape_1_tensor).BindOutput(transpose_1_tensor);
  (*perm_op_2).BindInput(transpose_1_tensor).BindOutput(transpose_2_tensor);
  (*reshape_op_2).BindInput(transpose_2_tensor).BindOutput(reshape_2_tensor);

  if ((reshape_2_shape[0] > pad_infer_shape[0]) ||
      (reshape_2_shape[1] > pad_infer_shape[1])) {
    //has pad inside
    //slice back and bottom

    // pad_infer_shape = {480, 480, in_channel, 1}; //resnet quant model
    auto slice_spec = in_spec.SetShape(pad_infer_shape);

    auto slice_tensor = context_->batch_fuse_graph_->CreateTensor(slice_spec);
    std::vector<int32_t> length = {(int32_t)pad_infer_shape[0],
                                   (int32_t)pad_infer_shape[1],
                                   (int32_t)in_channel, 1};
    std::vector<int32_t> start = {0, 0, 0, 0};
    auto slice_op =
        context_->batch_fuse_graph_->CreateOperation<vx::ops::Slice>(0, start,
                                                                     length);
    (*slice_op).BindInput(reshape_2_tensor).BindOutput(slice_tensor);
    return slice_tensor;
  }

  return reshape_2_tensor;
}

#define CREATE_AND_CONCAT_OP(idx, start, length)                             \
  \              
  auto idx##_op =                                                            \
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Slice>(0, start, \
                                                                   length);  \
  vx::ShapeType idx##_shape({out_w, out_h, out_channel, 1});                 \
  auto idx##_spec = input_spec.SetShape(idx##_shape);                        \
  auto idx##_tensor = context_->batch_fuse_graph_->CreateTensor(idx##_spec); \
  (*idx##_op).BindInput(input).BindOutput(idx##_tensor);                     \
  tensors.push_back(idx##_tensor);

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertSliceAndConcat(
    std::shared_ptr<vx::Tensor> input, bool is_graph_output,
    std::shared_ptr<vx::Tensor> src_out) {
  auto input_spec = input->GetSpec();
  auto input_shape = input->GetShape();  //    bw bh c1
  auto t_src = src_out;  //context_->GetBatchFuseMapedTensor(input);
  auto t_src_shape = t_src->GetShape();  // whcn
  auto out_spec = t_src->GetSpec();
  uint32_t batch = t_src_shape[3];
  uint32_t batch_factor_w = ClosestFactors(t_src_shape[3]).first;
  uint32_t batch_factor_h = ClosestFactors(t_src_shape[3]).second;
  // uint32_t sqrt_batch = sqrt(batch);
  uint32_t batch_out_h = input_shape[1];
  uint32_t batch_out_w = input_shape[0];
  uint32_t out_h = t_src_shape[1];
  uint32_t out_w = t_src_shape[0];
  uint32_t out_channel = t_src_shape[2];

  //if there has shared pad between valid value, overlap size may be negative
  int32_t overlap_h = 0;
  int32_t overlap_w = 0;
  if (batch_factor_h - 1 == 0)
    overlap_h = 0;
  else
    overlap_h = (batch_out_h - batch_factor_h * out_h) / (batch_factor_h - 1);
  if (batch_factor_w - 1 == 0)
    overlap_w = 0;
  else
    overlap_w = (batch_out_w - batch_factor_w * out_w) / (batch_factor_w - 1);

  int32_t out_w_ = static_cast<int32_t>(out_w);
  int32_t out_h_ = static_cast<int32_t>(out_h);
  int32_t out_channel_ = static_cast<int32_t>(out_channel);

  std::vector<int32_t> axis_point_h(batch_factor_h, 0);
  std::vector<int32_t> axis_point_w(batch_factor_w, 0);

  std::vector<int32_t> length = {out_w_, out_h_, out_channel_, 1};
  std::vector<std::vector<int32_t>> start_point;

  for (int i = 0; i < batch_factor_h; i++) {
    axis_point_h[i] = 0 + i * (overlap_h + out_h);
  }

  for (int i = 0; i < batch_factor_w; i++) {
    axis_point_w[i] = 0 + i * (overlap_w + out_w);
  }

  for (int i = 0; i < batch_factor_h; i++) {
    for (int j = 0; j < batch_factor_w; j++) {
      start_point.push_back({axis_point_w[j], axis_point_h[i], 0, 0});
    }
  }

  std::vector<std::shared_ptr<vx::Tensor>> tensors;
  for (int i = 0; i < batch; i++) {
    CREATE_AND_CONCAT_OP(i, start_point[i], length);
  }
  auto slice_shape = tensors[0]->GetSpec();

  auto concat =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Concat>(3, batch);
  auto concat_tensor = context_->batch_fuse_graph_->CreateTensor(out_spec);
  auto concat_shape = concat_tensor->GetShape();
  (*concat).BindInputs(tensors).BindOutput(concat_tensor);
  return concat_tensor;
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertMask(
    std::shared_ptr<vx::Tensor> input, bool is_graph_output,
    std::shared_ptr<vx::Tensor> src_in) {
  auto input_spec = input->GetSpec();    // whcn
  auto input_shape = input->GetShape();  // whcn
  auto src_shape = src_in->GetShape();
  uint32_t batch = src_shape[3];
  uint32_t batch_factor_w = ClosestFactors(batch).first;
  uint32_t batch_factor_h = ClosestFactors(batch).second;

  uint32_t batch_out_h = input_shape[1];
  uint32_t batch_out_w = input_shape[0];
  uint32_t out_h = src_shape[1];
  uint32_t out_w = src_shape[0];
  uint32_t in_channel = src_shape[2];

  //if there has shared pad between valid value, overlap size may be negative
  int32_t overlap_h = 0;
  int32_t overlap_w = 0;
  if (batch_factor_h - 1 == 0)
    overlap_h = 0;
  else
    overlap_h = (batch_out_h - batch_factor_h * out_h) / (batch_factor_h - 1);
  if (batch_factor_w - 1 == 0)
    overlap_w = 0;
  else
    overlap_w = (batch_out_w - batch_factor_w * out_w) / (batch_factor_w - 1);

  if (overlap_h < 0 && overlap_w < 0) {
    return input;
  }

  std::vector<uint32_t> axis_point_h(batch_factor_h, 0);
  std::vector<uint32_t> axis_point_w(batch_factor_w, 0);
  std::vector<std::vector<uint32_t>> start_point;

  for (int i = 0; i < batch_factor_h; i++) {
    axis_point_h[i] = 0 + i * (overlap_h + out_h);
  }

  for (int i = 0; i < batch_factor_w; i++) {
    axis_point_w[i] = 0 + i * (overlap_w + out_w);
  }

  for (int i = 0; i < batch_factor_w; i++) {
    for (int j = 0; j < batch_factor_h; j++) {
      start_point.push_back({axis_point_w[i], axis_point_h[j], 0, 0});
    }
  }
  std::vector<uint32_t> length = {out_w, out_h, in_channel, 1};

  std::vector<std::vector<std::vector<uint8_t>>> mask_data(
      in_channel, std::vector<std::vector<uint8_t>>(
                      batch_out_h, std::vector<uint8_t>(batch_out_w, 0)));

  for (int i = 0; i < start_point.size(); i++) {
    int w_start = start_point[i][0];
    int h_start = start_point[i][1];
    int c_start = start_point[i][2];
    for (int c = 0; c < in_channel; c++) {
      for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
          mask_data[c_start + c][h_start + h][w_start + w] = 1;
        }
      }
    }
  }

  std::vector<uint8_t> mask_vector;
  for (int c = 0; c < in_channel; c++) {
    for (int h = 0; h < batch_out_h; h++) {
      for (int w = 0; w < batch_out_w; w++) {
        mask_vector.push_back(mask_data[c][h][w]);
      }
    }
  }
  float scales = 1;
  int zp = 0;

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, scales, zp);
  vx::TensorSpec mask_spec(input_spec.datatype_, input_shape,
                           tim::vx::TensorAttribute::CONSTANT, quant_input);
  auto mask_tensor =
      context_->batch_fuse_graph_->CreateTensor(mask_spec, mask_vector.data());


  if (input_spec.datatype_ == vx::DataType::FLOAT32) {
    std::vector<std::vector<std::vector<float>>> mask_data_float(
        in_channel, std::vector<std::vector<float>>(
                        batch_out_h, std::vector<float>(batch_out_w, 0)));

    for (int i = 0; i < start_point.size(); i++) {
      int w_start = start_point[i][0];
      int h_start = start_point[i][1];
      int c_start = start_point[i][2];
      for (int c = 0; c < in_channel; c++) {
        for (int h = 0; h < out_h; h++) {
          for (int w = 0; w < out_w; w++) {
            mask_data_float[c_start + c][h_start + h][w_start + w] = 1;
          }
        }
      }
    }

    std::vector<float> mask_vector_float;
    for (int c = 0; c < in_channel; c++) {
      for (int h = 0; h < batch_out_h; h++) {
        for (int w = 0; w < batch_out_w; w++) {
          mask_vector_float.push_back(mask_data[c][h][w]);
        }
      }
    }
    
    vx::TensorSpec mask_spec(input_spec.datatype_, input_shape,
                             tim::vx::TensorAttribute::CONSTANT);

    mask_tensor =
      context_->batch_fuse_graph_->CreateTensor(mask_spec, mask_vector_float.data());
  }

  
  auto mask_out = context_->batch_fuse_graph_->CreateTensor(input_spec);
  auto mask_op =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Multiply>();
  (*mask_op).BindInputs({input, mask_tensor}).BindOutput(mask_out);
  return mask_out;
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertPad(
    std::shared_ptr<vx::Tensor> input, bool is_graph_output,
    std::shared_ptr<vx::Tensor> src_in) {
  auto input_spec = input->GetSpec();    // whcn
  auto input_shape = input->GetShape();  // whcn

  uint32_t pad_h = 0;
  uint32_t pad_w = 0;
  uint32_t out_h = input_shape[1];
  uint32_t out_w = input_shape[0];

  ////aanother way to compute pad size
  // uint32_t batch_factor_w = ClosestFactors(input_shape[3]).first;
  // uint32_t batch_factor_h = ClosestFactors(input_shape[3]).second;
  // auto pad_infer_shape =
  //     context_->GetPadInferShape(input);  //inside paded tensor
  // if (batch_factor_h - 1 == 0)
  //   pad_h = 0;
  // else
  //   pad_h =
  //       (pad_infer_shape[0] - batch_factor_h * out_h) / (batch_factor_h - 1);
  // if (batch_factor_w - 1 == 0)
  //   pad_w = 0;
  // else
  //   pad_w =
  //       (pad_infer_shape[1] - batch_factor_w * out_w) / (batch_factor_w - 1);

  auto pad = context_->GetForwardPad(src_in);
  if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0) {
    // no pad
    return input;
  }
  std::vector<uint32_t> front_size = {0, 0, 0, 0};
  // std::vector<uint32_t> back_size = {32, 32, 0, 0};//resnet quant model
  std::vector<uint32_t> back_size = {pad[1], pad[3], 0, 0};
  auto pad_op = context_->batch_fuse_graph_->CreateOperation<vx::ops::Pad>(
      front_size, back_size, 0, tim::vx::ops::Pad::PAD_MODE_CONSTANT);

  vx::ShapeType pad_shape = {out_w + back_size[0], out_h + back_size[1],
                             input_shape[2], input_shape[3]};
  auto pad_spec = input_spec.SetShape(pad_shape);
  // vx::TensorSpec pad_spec(input_spec.datatype_, pad_shape,
  //                         tim::vx::TensorAttribute::TRANSIENT);
  auto pad_tensor = context_->batch_fuse_graph_->CreateTensor(pad_spec);
  (*pad_op).BindInput(input).BindOutput(pad_tensor);
  return pad_tensor;
}

std::vector<std::shared_ptr<vx::Tensor>> OpBatchFuse::CreateOutputsTensor() {
  std::vector<std::shared_ptr<vx::Tensor>> outputs_tensor;

  uint32_t i = 0;
  for (const auto& o : op_->impl()->OutputsTensor()) {
    auto out_shape = o->GetShape();
    auto out_spec = o->GetSpec();
    auto t_batch_fuse = context_->batch_fuse_graph_->CreateTensor(out_spec);
    context_->UpdateTensorMap(o, t_batch_fuse);
    context_->UpdateTensorBatchFuseMap(t_batch_fuse, o);
    outputs_tensor.push_back(t_batch_fuse);
    i++;
  }
  return outputs_tensor;
}

void OpBatchFuse::OnOutputs(
    std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) {
  auto graph_outputs = context_->clone_batch_graph_->OutputsTensor();
  auto op_outputs = op_->impl()->OutputsTensor();
  auto op_inputs = op_->impl()->InputsTensor();
  for (const auto& out : op_outputs) {
    if (graph_outputs.end() !=
        std::find(graph_outputs.begin(), graph_outputs.end(), out)) {
      context_->UpdateGraphOutputMap(out, context_->GetMapedTensor(out));
      // context_->UpdateGraphOutputMap(context_->GetMapedTensor(out), context_->GetMapedTensor(out));
      auto out_batch_fuse = context_->GetMapedTensor(out);
      auto out_batch_fuse_shape = out_batch_fuse->GetShape();
      uint32_t batch = out_batch_fuse_shape[out_batch_fuse_shape.size() - 1];
      auto out_shape = out->GetShape();
      uint32_t batch_src = out_shape[out_shape.size() - 1];

      // if (op_->impl()->kind_ !=2){
      //   auto slice_and_concat_out = InsertSliceAndConcat(context_->GetMapedTensor(out), true, out);
      //   auto slice_and_concat_out_shape = slice_and_concat_out->GetShape();
      //   context_->UpdateTensorMap(out, slice_and_concat_out);
      //   context_->UpdateTensorBatchFuseMap(slice_and_concat_out, out);
      //   context_->UpdateGraphOutputMap(out, slice_and_concat_out);
      // }

      if (context_->clone_batch_graph_->GetConsumersOp(out).empty()) {
        // The tensor is output of graph, and is not the input of other operations
        auto it = std::find(next_tensors.begin(), next_tensors.end(), out);
        if (it != next_tensors.end()) {
          next_tensors.erase(it);
        }
        if (batch == 1 && batch_src != 1) {
          auto slice_and_concat_out =
              InsertSliceAndConcat(context_->GetMapedTensor(out), true, out);
          auto slice_and_concat_out_shape = slice_and_concat_out->GetShape();
          context_->UpdateTensorMap(out, slice_and_concat_out);
          context_->UpdateTensorBatchFuseMap(slice_and_concat_out, out);
          context_->UpdateGraphOutputMap(out, slice_and_concat_out);
        }
      }
    }
  }
}

void OpBatchFuse::CloneGraph(
    std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) {
  auto op_outputs = op_->impl()->OutputsTensor();
  auto op_inputs = op_->impl()->InputsTensor();
  auto fake_batch = context_->GetFakeBatch();
  auto new_op = op_->Clone(context_->clone_batch_graph_);
  std::shared_ptr<vx::Tensor> clone_in_tensor;
  std::shared_ptr<vx::Tensor> clone_out_tensor;
  if (op_->impl()->kind_ == 2) {
    auto input_tensor = op_inputs[0];
    auto weight_tensor = op_inputs[1];
    auto bias_tensor = op_inputs[2];
    auto weight_spec = weight_tensor->GetSpec();  //whio
    auto bias_spec = bias_tensor->GetSpec();
    auto input_spec = input_tensor->GetSpec();
    std::vector<uint8_t> tmp_weight(weight_tensor->GetSpec().GetByteSize());
    std::vector<uint8_t> tmp_bias(bias_tensor->GetSpec().GetByteSize());
    weight_tensor->CopyDataFromTensor(tmp_weight.data());
    bias_tensor->CopyDataFromTensor(tmp_bias.data());
    auto clone_weight_tensor = context_->clone_batch_graph_->CreateTensor(
        weight_spec, tmp_weight.data());
    auto clone_bias_tensor =
        context_->clone_batch_graph_->CreateTensor(bias_spec, tmp_bias.data());
    auto clone_input_tensor = context_->GetCloneMapedTensor(input_tensor);
    (*new_op).BindInputs(
        {clone_input_tensor, clone_weight_tensor, clone_bias_tensor});

    auto out_shape = op_outputs[0]->GetShape();
    auto out_spec = op_outputs[0]->GetSpec();
    auto batch = out_shape[3];
    vx::ShapeType new_shape = {out_shape[1], out_shape[2], out_shape[0],
                               fake_batch};
    // vx::TensorSpec new_spec(out_spec.datatype_, new_shape, out_spec.attr_);
    auto new_spec = out_spec.SetShape(new_shape);
    auto clone_out_tensor =
        context_->clone_batch_graph_->CreateTensor(new_spec);
    context_->UpdateCloneTensorMap(op_outputs[0], clone_out_tensor);
    (*new_op).BindOutput(clone_out_tensor);
    if (out_spec.attr_ == vx::TensorAttribute::OUTPUT) {
      context_->UpdateGraphOutputMap(op_outputs[0], clone_out_tensor);
    }
    if (input_spec.attr_ == vx::TensorAttribute::INPUT) {
      context_->UpdateGraphOutputMap(input_tensor, clone_input_tensor);
    }

  } else {
    //input
    for (auto input : op_inputs) {
      if (input->GetSpec().attr_ == vx::TensorAttribute::INPUT) {
        //graph input
        auto in_shape = input->GetShape();
        auto in_spec = input->GetSpec();
        auto batch = in_shape[3];
        vx::ShapeType new_shape = {in_shape[0], in_shape[1], in_shape[2],
                                   fake_batch};
        // vx::TensorSpec new_spec(in_spec.datatype_, new_shape, in_spec.attr_);
        auto new_spec = in_spec.SetShape(new_shape);
        clone_in_tensor = context_->clone_batch_graph_->CreateTensor(new_spec);
        context_->UpdateGraphInputMap(input, clone_in_tensor);
      } else {
        clone_in_tensor = context_->GetCloneMapedTensor(input);
      }
      context_->UpdateCloneTensorMap(input, clone_in_tensor);
      (*new_op).BindInput(clone_in_tensor);
    }

    //output
    auto in_shape = op_inputs[0]->GetShape();
    auto in_spec = op_inputs[0]->GetSpec();
    auto out_shape = op_outputs[0]->GetShape();
    auto out_spec = op_outputs[0]->GetSpec();
    vx::ShapeType new_shape;
    if ((op_->impl()->kind_ == 19) &&
        (op_inputs[0]->GetSpec().attr_ != vx::TensorAttribute::INPUT)) {
      //transpose op
      new_shape = {in_shape[0], in_shape[1], in_shape[2], fake_batch};
    } else if (op_->impl()->kind_ == 162) {
      //the last reshape
      new_shape = {out_shape[0], fake_batch};
    } else {
      new_shape = {out_shape[1], out_shape[2], out_shape[0], fake_batch};
    }
    // vx::TensorSpec new_spec(out_spec.datatype_, new_shape, out_spec.attr_);
    auto new_spec = out_spec.SetShape(new_shape);
    clone_out_tensor = context_->clone_batch_graph_->CreateTensor(new_spec);
    context_->UpdateCloneTensorMap(op_outputs[0], clone_out_tensor);
    (*new_op).BindOutput(clone_out_tensor);
    if (out_spec.attr_ == vx::TensorAttribute::OUTPUT) {
      context_->UpdateGraphOutputMap(op_outputs[0], clone_out_tensor);
    }
  }
  next_tensors.push_back(op_outputs[0]);
}

vx::PadType OpBatchFuse::TranslatePadType(int32_t pad) {
  switch (pad) {
    case VSI_NN_PAD_AUTO:
      return vx::PadType::AUTO;
    case VSI_NN_PAD_VALID:
      return vx::PadType::VALID;
    case VSI_NN_PAD_SAME:
      return vx::PadType::SAME;
    default:
      return vx::PadType::AUTO;
  }
}

vx::PoolType OpBatchFuse::TranslatePoolType(int32_t pool) {
  switch (pool) {
    case VX_CONVOLUTIONAL_NETWORK_POOLING_MAX:
      return vx::PoolType::MAX;
    case VX_CONVOLUTIONAL_NETWORK_POOLING_AVG:
      return vx::PoolType::AVG;
    case VX_CONVOLUTIONAL_NETWORK_POOLING_L2:
      return vx::PoolType::L2;
    case VX_CONVOLUTIONAL_NETWORK_POOLING_AVG_ANDROID:
      return vx::PoolType::AVG_ANDROID;
    default:
      return vx::PoolType::MAX;
  }
}

vx::RoundType OpBatchFuse::TranslateRoundType(int32_t round) {
  switch (round) {
    case VSI_NN_ROUND_CEIL:
      return vx::RoundType::CEILING;
    case VSI_NN_ROUND_FLOOR:
      return vx::RoundType::FLOOR;
    default:
      return vx::RoundType::FLOOR;
  }
}

}  // namespace fuse

}  // namespace tim