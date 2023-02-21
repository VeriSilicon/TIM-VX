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
#include "builtin_op_impl.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/pad.h"
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

std::vector<std::vector<int32_t>> OpBatchFuse::ComputeStartPoints(
    std::vector<uint32_t> input_batch_fuse_shape,
    std::vector<uint32_t> input_shape, uint32_t batch_axis,
    std::vector<uint32_t> fuse_axes) {
  auto w_axis = fuse_axes[0];
  auto h_axis = fuse_axes[1];

  uint32_t batch = input_shape[batch_axis];
  uint32_t batch_factor_w = ClosestFactors(batch).first;
  uint32_t batch_factor_h = ClosestFactors(batch).second;

  uint32_t batch_out_h = input_batch_fuse_shape[h_axis];
  uint32_t batch_out_w = input_batch_fuse_shape[w_axis];
  uint32_t out_h = input_shape[h_axis];
  uint32_t out_w = input_shape[w_axis];

  //if there has shared pad between valid value, overlap size may be negative
  int32_t gap_output_h = 0;
  int32_t gap_output_w = 0;
  if (batch_factor_h - 1 == 0)
    gap_output_h = 0;
  else
    gap_output_h =
        (batch_out_h - batch_factor_h * out_h) / (batch_factor_h - 1);
  if (batch_factor_w - 1 == 0)
    gap_output_w = 0;
  else
    gap_output_w =
        (batch_out_w - batch_factor_w * out_w) / (batch_factor_w - 1);

  std::vector<int32_t> axis_point_h(batch_factor_h, 0);
  std::vector<int32_t> axis_point_w(batch_factor_w, 0);

  std::vector<std::vector<int32_t>> start_point;

  // Compute the start point of each piece of data
  for (uint i = 0; i < batch_factor_h; i++) {
    axis_point_h[i] = 0 + i * (gap_output_h + out_h);
  }

  for (uint i = 0; i < batch_factor_w; i++) {
    axis_point_w[i] = 0 + i * (gap_output_w + out_w);
  }

  for (uint i = 0; i < batch_factor_h; i++) {
    for (uint j = 0; j < batch_factor_w; j++) {
      std::vector<int32_t> point(4, 0);
      point[w_axis] = axis_point_w[j];
      point[h_axis] = axis_point_h[i];
      start_point.push_back(point);
    }
  }

  return start_point;
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertPermuteAndReshape(
    std::shared_ptr<vx::Tensor> pad_tensor,
    std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
    std::vector<uint32_t> fuse_axes) {
  // pad_tensor: belong to batch fused graph, has not been fused
  // input_tensor: belong to source graph
  auto pad_spec = pad_tensor->GetSpec();    // whcn
  auto pad_shape = pad_tensor->GetShape();  // whcn
  auto gap_infer_shape =
      context_->GetGapInferShape(input_tensor);  //inside paded tensor
  uint32_t batch_factor_w = ClosestFactors(pad_shape[batch_axis]).first;
  uint32_t batch_factor_h = ClosestFactors(pad_shape[batch_axis]).second;
  auto w_axis = fuse_axes[0];
  auto h_axis = fuse_axes[1];
  auto c_axis = context_->GetPermChannelAxis(input_tensor);
  uint32_t in_channel = pad_shape[c_axis];
  uint32_t h = pad_shape[h_axis];
  uint32_t w = pad_shape[w_axis];

  // Batch fuse through tanspose
  if (w_axis != 0 && h_axis != 1 && c_axis != 2) {
    VSILOGE("Only supports shape as WHCN instead of CWHN");
  }

  // The following parameters and shapes are only applicable to tensors whose shape is WHCN
  // According to the rules of layout inference, the shape of ops such as conv2d and pool2d will be transferred to WHCN,
  // and these ops need batch fuse, so the shape of CWHN will not appear here logically,
  // so only an error log is added here, not according to axis to map the shape.

  std::vector<uint32_t> perm1 = {0, 1, 3, 4, 2, 5};
  std::vector<uint32_t> perm2 = {0, 2, 1, 3, 4, 5};
  vx::ShapeType reshape_1_shape(
      {w, h, in_channel, batch_factor_w, batch_factor_h, 1});  //whcbbn
  vx::ShapeType transpose_1_shape(
      {w, h, batch_factor_w, batch_factor_h, in_channel, 1});  //whbbcn
  vx::ShapeType transpose_2_shape(
      {w, batch_factor_w, h, batch_factor_h, in_channel, 1});  //wbhbcn
  vx::ShapeType reshape_2_shape(
      {w * batch_factor_w, h * batch_factor_h, in_channel, 1});  //whcn

  auto reshape_1_spec = pad_spec.SetShape(reshape_1_shape);
  auto transpose_1_spec = pad_spec.SetShape(transpose_1_shape);
  auto transpose_2_spec = pad_spec.SetShape(transpose_2_shape);
  auto reshape_2_spec = pad_spec.SetShape(reshape_2_shape);

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

  (*reshape_op_1).BindInput(pad_tensor).BindOutput(reshape_1_tensor);
  (*perm_op_1).BindInput(reshape_1_tensor).BindOutput(transpose_1_tensor);
  (*perm_op_2).BindInput(transpose_1_tensor).BindOutput(transpose_2_tensor);
  (*reshape_op_2).BindInput(transpose_2_tensor).BindOutput(reshape_2_tensor);

  if ((reshape_2_shape[w_axis] > gap_infer_shape[w_axis]) ||
      (reshape_2_shape[h_axis] > gap_infer_shape[h_axis])) {
    // In genernal, before fuse, a tensor will be pad first, so after batch fuse,
    // there are extra pixels on fused tensor's right and bottom, so it need slice
    // to the gap_infer_shape.

    // TODO(HuanyuCai): There are some problems in gap infer shape map, the maped shape may be different
    // to the shape which is computed by OnInputs()

    auto slice_spec = pad_spec.SetShape(gap_infer_shape);
    auto slice_tensor = context_->batch_fuse_graph_->CreateTensor(slice_spec);
    std::vector<int32_t> length = {0, 0, 0, 0};
    length[w_axis] = (int32_t)gap_infer_shape[w_axis];
    length[h_axis] = (int32_t)gap_infer_shape[h_axis];
    length[c_axis] = (int32_t)in_channel;
    length[batch_axis] = 1;
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
  idx##_shape[w_axis] = out_w;                                               \
  idx##_shape[h_axis] = out_h;                                               \
  idx##_shape[c_axis] = out_channel;                                         \
  idx##_shape[batch_axis] = 1;                                               \
  auto idx##_spec = input_batch_fuse_spec.SetShape(idx##_shape);             \
  auto idx##_tensor = context_->batch_fuse_graph_->CreateTensor(idx##_spec); \
  (*idx##_op).BindInput(input_batch_fuse_tensor).BindOutput(idx##_tensor);   \
  tensors.push_back(idx##_tensor);

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertSliceAndConcat(
    std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
    std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
    std::vector<uint32_t> fuse_axes) {
  // input_batch_fuse: batch fused tensor, n == 1
  // input_tensor: belong to source graph n >= 1
  // Slice batch fused tensor

  auto input_spec = input_tensor->GetSpec();
  auto input_shape = input_tensor->GetShape();
  auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();  // whcn
  auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();

  auto w_axis = fuse_axes[0];
  auto h_axis = fuse_axes[1];
  auto c_axis = context_->GetPermChannelAxis(input_tensor);

  uint32_t out_h = input_shape[h_axis];
  uint32_t out_w = input_shape[w_axis];
  uint32_t out_channel = input_shape[c_axis];

  int32_t out_w_ = static_cast<int32_t>(out_w);
  int32_t out_h_ = static_cast<int32_t>(out_h);
  int32_t out_channel_ = static_cast<int32_t>(out_channel);

  std::vector<int32_t> length(4);
  length[w_axis] = out_w_;
  length[h_axis] = out_h_;
  length[c_axis] = out_channel_;
  length[batch_axis] = 1;

  auto start_point = ComputeStartPoints(input_batch_fuse_shape, input_shape,
                                        batch_axis, fuse_axes);
  uint32_t batch = input_shape[batch_axis];

  std::vector<std::shared_ptr<vx::Tensor>> tensors;
  for (uint i = 0; i < batch; i++) {
    CREATE_AND_CONCAT_OP(i, start_point[i], length);
  }
  auto slice_shape = tensors[0]->GetSpec();

  auto concat = context_->batch_fuse_graph_->CreateOperation<vx::ops::Concat>(
      batch_axis, batch);

  // Concat tensor has the same spec of original tensor
  auto concat_tensor = context_->batch_fuse_graph_->CreateTensor(input_spec);
  auto concat_shape = concat_tensor->GetShape();
  (*concat).BindInputs(tensors).BindOutput(concat_tensor);
  return concat_tensor;
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertMask(
    std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
    std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
    std::vector<uint32_t> fuse_axes) {
  // input_batch_fuse_tensor: belong to batch fuse graph
  // input_tensor: belong to source graph
  auto input_spec = input_tensor->GetSpec();
  auto input_shape = input_tensor->GetShape();
  auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();  // whcn
  auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();

  auto w_axis = fuse_axes[0];
  auto h_axis = fuse_axes[1];
  auto c_axis = context_->GetPermChannelAxis(input_tensor);

  uint32_t out_h = input_shape[h_axis];
  uint32_t out_w = input_shape[w_axis];
  uint32_t out_channel = input_shape[c_axis];
  uint32_t batch_out_h = input_batch_fuse_shape[h_axis];
  uint32_t batch_out_w = input_batch_fuse_shape[w_axis];

  auto start_point = ComputeStartPoints(input_batch_fuse_shape, input_shape,
                                        batch_axis, fuse_axes);

  std::shared_ptr<vx::Tensor> mask_tensor;
  std::vector<uint32_t> index(3);
  index[2 - w_axis] = batch_out_w;
  index[2 - h_axis] = batch_out_h;
  index[2 - c_axis] = out_channel;

  if (input_spec.datatype_ == vx::DataType::FLOAT32) {
    std::vector<std::vector<std::vector<float>>> mask_data_float(
        index[0], std::vector<std::vector<float>>(
                      index[1], std::vector<float>(index[2], 0)));

    // Set valid area with mask value 1
    // Set gap area with mask value 0

    for (uint i = 0; i < start_point.size(); i++) {
      int w_start = start_point[i][w_axis];
      int h_start = start_point[i][h_axis];
      int c_start = start_point[i][c_axis];
      std::vector<uint32_t> index_(3);
      for (uint c = 0; c < out_channel; c++) {
        for (uint h = 0; h < out_h; h++) {
          for (uint w = 0; w < out_w; w++) {
            index_[2 - w_axis] = w_start + w;
            index_[2 - h_axis] = h_start + h;
            index_[2 - c_axis] = c_start + c;
            mask_data_float[index_[0]][index_[1]][index_[2]] = 1;
          }
        }
      }
    }

    std::vector<float> mask_vector_float;
    std::vector<uint32_t> index_(3);
    for (uint c = 0; c < out_channel; c++) {
      for (uint h = 0; h < batch_out_h; h++) {
        for (uint w = 0; w < batch_out_w; w++) {
          index_[2 - w_axis] = w;
          index_[2 - h_axis] = h;
          index_[2 - c_axis] = c;
          mask_vector_float.push_back(
              mask_data_float[index_[0]][index_[1]][index_[2]]);
        }
      }
    }

    vx::TensorSpec mask_spec(input_batch_fuse_spec.datatype_,
                             input_batch_fuse_shape,
                             tim::vx::TensorAttribute::CONSTANT);

    mask_tensor = context_->batch_fuse_graph_->CreateTensor(
        mask_spec, mask_vector_float.data());

  } else if (input_spec.datatype_ == vx::DataType::UINT8) {
    std::vector<std::vector<std::vector<float>>> mask_data(
        index[0], std::vector<std::vector<float>>(
                      index[1], std::vector<float>(index[2], 0)));

    for (uint i = 0; i < start_point.size(); i++) {
      int w_start = start_point[i][w_axis];
      int h_start = start_point[i][h_axis];
      int c_start = start_point[i][c_axis];
      std::vector<uint32_t> index_(3);
      for (uint c = 0; c < out_channel; c++) {
        for (uint h = 0; h < out_h; h++) {
          for (uint w = 0; w < out_w; w++) {
            index_[2 - w_axis] = w_start + w;
            index_[2 - h_axis] = h_start + h;
            index_[2 - c_axis] = c_start + c;
            mask_data[index_[0]][index_[1]][index_[2]] = 1;
          }
        }
      }
    }

    std::vector<uint8_t> mask_vector;
    std::vector<uint32_t> index_(3);
    for (uint c = 0; c < out_channel; c++) {
      for (uint h = 0; h < batch_out_h; h++) {
        for (uint w = 0; w < batch_out_w; w++) {
          index_[2 - w_axis] = w;
          index_[2 - h_axis] = h;
          index_[2 - c_axis] = c;
          mask_vector.push_back(mask_data[index_[0]][index_[1]][index_[2]]);
        }
      }
    }

    // Set mask tensor's scales = 1 and zp = 0 to mask batch fused tensor correctly
    float scales = 1;
    int zp = 0;
    tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, scales,
                                      zp);
    vx::TensorSpec mask_spec(input_batch_fuse_spec.datatype_,
                             input_batch_fuse_shape,
                             tim::vx::TensorAttribute::CONSTANT, quant_input);
    mask_tensor = context_->batch_fuse_graph_->CreateTensor(mask_spec,
                                                            mask_vector.data());
  } else {
    VSILOGE("No Support for this data type");
    // TODO(HuanyuCai): support other data type and make code smarter when initialize vector with different data type
  }

  auto mask_out =
      context_->batch_fuse_graph_->CreateTensor(input_batch_fuse_spec);
  auto mask_op =
      context_->batch_fuse_graph_->CreateOperation<vx::ops::Multiply>();
  (*mask_op)
      .BindInputs({input_batch_fuse_tensor, mask_tensor})
      .BindOutput(mask_out);
  return mask_out;
}

std::shared_ptr<vx::Tensor> OpBatchFuse::InsertPad(
    std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
    std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
    std::vector<uint32_t> fuse_axes) {
  // input_batch_fuse_tensor: belong to batch fuse graph
  // input_tensor: belong to source graph

  auto input_batch_fuse_spec = input_batch_fuse_tensor->GetSpec();    // whcn
  auto input_batch_fuse_shape = input_batch_fuse_tensor->GetShape();  // whcn
  auto w_axis = fuse_axes[0];
  auto h_axis = fuse_axes[1];

  uint32_t out_w = input_batch_fuse_shape[w_axis];
  uint32_t out_h = input_batch_fuse_shape[h_axis];

  auto gap = context_->GetForwardGap(input_tensor);
  if (gap[w_axis] == 0 && gap[h_axis] == 0) {
    return input_batch_fuse_tensor;
  }
  // Pad tensor on w & h axis
  std::vector<uint32_t> front_size = {0, 0, 0, 0};
  std::vector<uint32_t> back_size = {0, 0, 0, 0};
  back_size[w_axis] = gap[w_axis];
  back_size[h_axis] = gap[h_axis];
  auto pad_op = context_->batch_fuse_graph_->CreateOperation<vx::ops::Pad>(
      front_size, back_size, 0, tim::vx::ops::Pad::PAD_MODE_CONSTANT);

  vx::ShapeType pad_shape = {0, 0, 0, 0};
  pad_shape[w_axis] = out_w + back_size[w_axis];
  pad_shape[h_axis] = out_h + back_size[h_axis];
  pad_shape[batch_axis] = input_batch_fuse_shape[batch_axis];
  for (uint32_t i(0); i < pad_shape.size(); i++) {
    if (i != w_axis && i != h_axis && i != batch_axis) {
      pad_shape[i] = input_batch_fuse_shape[i];
    }
  }
  auto pad_spec = input_batch_fuse_spec.SetShape(pad_shape);
  auto pad_tensor = context_->batch_fuse_graph_->CreateTensor(pad_spec);
  (*pad_op).BindInput(input_batch_fuse_tensor).BindOutput(pad_tensor);
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
      // Out tensor is output of graph
      auto out_batch_fuse = context_->GetMapedTensor(out);
      auto out_batch_fuse_shape = out_batch_fuse->GetShape();

      // auto batch_src_axis = context_->GetBatchAxis();  // 3
      auto fuse_src_axes = context_->GetFuseAxes();  // [1, 2]

      auto perm_axis_map = context_->GetPermAxisMap(out);
      auto fuse_axes = context_->GetPermFuseAxes(out);
      auto batch_axis = context_->GetPermBatchAxis(out);

      // auto w_axis = fuse_axes[0];
      // auto h_axis = fuse_axes[1];

      uint32_t batch = out_batch_fuse_shape[out_batch_fuse_shape.size() - 1];
      auto out_shape = out->GetShape();
      uint32_t batch_src = out_shape[out_shape.size() - 1];

      if (context_->clone_batch_graph_->GetConsumersOp(out).empty()) {
        // The tensor is output of graph, and is not the input of other operations
        auto it = std::find(next_tensors.begin(), next_tensors.end(), out);
        if (it != next_tensors.end()) {
          next_tensors.erase(it);
        }
        if (batch == 1 && batch_src != 1) {
          // If the tensor is output, it need slice and concat
          auto slice_and_concat_out = InsertSliceAndConcat(
              context_->GetMapedTensor(out), out, batch_axis, fuse_axes);
          auto slice_and_concat_out_shape = slice_and_concat_out->GetShape();
          context_->UpdateTensorMap(out, slice_and_concat_out);
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

  auto input_tensor = op_inputs[0];
  auto output_tensor = op_outputs[0];

  for (auto input : op_inputs) {
    auto input_shape = input->GetShape();
    auto input_spec = input->GetSpec();

    if (input_spec.attr_ == vx::TensorAttribute::INPUT) {
      // Set multi batch
      input_shape[input_shape.size() - 1] = fake_batch;
      input_spec.SetShape(input_shape);
      clone_in_tensor = context_->clone_batch_graph_->CreateTensor(input_spec);
      context_->UpdatePermAxisMap(clone_in_tensor, {0, 1, 2, 3});
    } else {
      if (op_->impl()->kind_ == 2 &&
          input_spec.attr_ == vx::TensorAttribute::CONSTANT) {
        // weight or bias

        std::vector<uint8_t> tmp(input_spec.GetByteSize());
        input->CopyDataFromTensor(tmp.data());

        // Original tensor spec
        clone_in_tensor = context_->clone_batch_graph_->CreateTensor(
            input->GetSpec(), tmp.data());

      } else {
        clone_in_tensor = context_->GetCloneMapedTensor(input);
      }
    }
    context_->UpdateCloneTensorMap(input, clone_in_tensor);
    (*new_op).BindInput(clone_in_tensor);
  }

  for (auto output : op_outputs) {
    auto output_shape = output->GetShape();
    auto output_spec = output->GetSpec();
    // Set multi batch
    output_shape[output_shape.size() - 1] = fake_batch;
    output_spec.SetShape(output_shape);
    clone_out_tensor = context_->clone_batch_graph_->CreateTensor(output_spec);
    context_->UpdateCloneTensorMap(output, clone_out_tensor);
    auto input_perm =
        context_->GetPermAxisMap(context_->GetCloneMapedTensor(input_tensor));
    if (op_->impl()->kind_ == 19) {
      // Transpose
      std::vector<uint32_t> perm(op_->impl()->node()->nn_param.permute.dim_num);
      memcpy(perm.data(), op_->impl()->node()->nn_param.permute.perm,
             op_->impl()->node()->nn_param.permute.dim_num * sizeof(uint32_t));
      std::vector<uint32_t> output_perm = {0, 1, 2, 3};
      output_perm[0] = input_perm[perm[0]];
      output_perm[1] = input_perm[perm[1]];
      output_perm[2] = input_perm[perm[2]];
      output_perm[3] = input_perm[perm[3]];
      context_->UpdatePermAxisMap(clone_out_tensor, output_perm);
    } else if (op_->impl()->kind_ == 162) {
      // Reshape
      // If the input shape is WHCN and output shape is not, how to define output tensor's perm? 
      // How to map the perm, what does each dimension obtained by map, and how does it affect the subsequent tensor?
      // Although map the perm axis is a concept that only applies to ops (conv2d, pool2d) with physical meanings in shape, 
      // we also need to assign a value that is as correct as possible for other ops such as reshape
      context_->UpdatePermAxisMap(clone_out_tensor, input_perm);
    } else {
      context_->UpdatePermAxisMap(clone_out_tensor, input_perm);
    }

    (*new_op).BindOutput(clone_out_tensor);
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