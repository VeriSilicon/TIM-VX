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

#include "op_layout_inference.h"
#include "src/tim/layout_infer/permute_vector.h"
#include "src/tim/vx/operation_private.h"
#include "tim/vx/ops/transpose.h"
#include "src/tim/vx/type_utils.h"

#include <algorithm>
#include <vector>

namespace tim {
namespace transform {
void OpLayoutInfer::OnOutputs(
    std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) {
  auto graph_outputs = context_->src_graph_->OutputsTensor();
  auto op_outputs = op_->impl()->OutputsTensor();
  for (const auto& out : op_outputs) {
    if (graph_outputs.end() !=
        std::find(graph_outputs.begin(), graph_outputs.end(), out)) {
      auto pv = context_->GetPermuteVector(out);
      if (!pv->IsAligned()) {
        auto perm_out = InsertPermute(context_->GetMapedTensor(out),
                                      pv->Reverse(), true, out);
        // Update graph out tensor
        context_->UpdateTensorMap(out, perm_out);
      }
      if (!context_->src_graph_->GetConsumersOp(out).empty()) {
        // The tensor is output of graph, but it also is the input of other operations
        context_->SetPermuteVector(out, MakeShared(pv->Rank()));
      } else {
        auto it = std::find(next_tensors.begin(), next_tensors.end(), out);
        if (it != next_tensors.end()) {
          next_tensors.erase(it);
        }
      }
    }
  }
}

std::shared_ptr<vx::Tensor> OpLayoutInfer::InsertPermute(
    std::shared_ptr<vx::Tensor> input, std::shared_ptr<IPermuteVector> perm,
    bool is_graph_output, std::shared_ptr<vx::Tensor> src_out) {
  auto out_spec = input->GetSpec();
  if (is_graph_output) {
    auto out_shape = src_out->GetShape();
    out_spec.SetShape(out_shape);
    out_spec.SetAttribute(vx::TensorAttribute::OUTPUT);
  } else {
    out_spec.SetAttribute(vx::TensorAttribute::TRANSIENT);
  }
  if (out_spec.quantization_.Type() == vx::QuantType::SYMMETRIC_PER_CHANNEL) {
    out_spec.quantization_.SetChannelDim(
        MapAxis(perm->AsStdVec(), out_spec.quantization_.ChannelDim()));
  }
  auto out_tensor = context_->infer_graph_->CreateTensor(out_spec);
  auto perm_op =
      context_->infer_graph_->CreateOperation<vx::ops::Transpose>(perm->AsStdVec());
  (*perm_op).BindInput(input).BindOutput(out_tensor);
  return out_tensor;
}

std::vector<std::shared_ptr<vx::Tensor>> OpLayoutInfer::CreateOutputsTensor(
    std::shared_ptr<IPermuteVector> required_pv) {
  std::vector<std::shared_ptr<vx::Tensor>> ouptuts_tensor;
  for (const auto& o : op_->impl()->OutputsTensor()) {
    auto in_shape = o->GetShape();
    auto out_spec = o->GetSpec();
    if (!required_pv->IsAligned()) {
      out_spec = out_spec.AsTransientSpec();
    }
    auto t_infer = context_->infer_graph_->CreateTensor(out_spec);
    context_->UpdateTensorMap(o, t_infer);
    ouptuts_tensor.push_back(t_infer);
  }
  return ouptuts_tensor;
}

vx::PadType OpLayoutInfer::TranslatePadType(int32_t pad) {
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

vx::PoolType OpLayoutInfer::TranslatePoolType(int32_t pool) {
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

vx::RoundType OpLayoutInfer::TranslateRoundType(int32_t round) {
  switch (round) {
    case VSI_NN_ROUND_CEIL:
      return vx::RoundType::CEILING;
    case VSI_NN_ROUND_FLOOR:
      return vx::RoundType::FLOOR;
    default:
      return vx::RoundType::FLOOR;
  }
}

uint32_t OpLayoutInfer::MapAxis(const std::vector<uint32_t>& perm,
                                uint32_t axis) {
  for (uint32_t i = 0; i < perm.size(); i++) {
    if (axis == perm[i]) {
      return i;
    }
  }
  VSILOGE("Map axis failed.");
  assert(false);
  return perm.size() - 1;
}

std::shared_ptr<IPermuteVector>
OpLayoutInfer::AlignPermuteVectorForMutilInputs() {
  auto src_inputs = op_->impl()->InputsTensor();
  // Suppose the inputs have same dimension rank
  // TODO(yzw): should choose a optimal required_pv
  auto required_pv = context_->GetPermuteVector(src_inputs[0]);
  for (const auto& i_src : src_inputs) {
    std::shared_ptr<vx::Tensor> perm_out;
    auto pv = context_->GetPermuteVector(i_src);
    auto final_pv = pv->Reverse()->Add(required_pv);
    if (!final_pv->IsAligned()) {
      if (i_src->IsConstTensor()) {
        perm_out = PermuteConstTensor(i_src, final_pv);
      } else {
        perm_out = InsertPermute(context_->GetMapedTensor(i_src), final_pv);
      }
      context_->UpdateTensorMap(i_src, perm_out);
      context_->SetPermuteVector(i_src, required_pv);
    }
  }
  return required_pv;
}

void OpLayoutInfer::ReverseInputsPermuteVector() {
  for (const auto& i_src : op_->impl()->InputsTensor()) {
    std::shared_ptr<vx::Tensor> perm_out;
    auto input_pv = context_->GetPermuteVector(i_src);
    if (!input_pv->IsAligned()) {
      if (i_src->IsConstTensor()) {
        perm_out = PermuteConstTensor(i_src, input_pv);
      } else {
        perm_out =
            InsertPermute(context_->GetMapedTensor(i_src), input_pv->Reverse());
      }
      context_->UpdateTensorMap(i_src, perm_out);
      context_->SetPermuteVector(i_src, MakeShared(input_pv->Rank()));
    }
  }
}

bool OpLayoutInfer::TransposeConstTensorData(
    const std::shared_ptr<vx::Tensor>& input,
    const std::shared_ptr<IPermuteVector>& pv, std::vector<uint8_t>& out_data) {
  auto vx_type = vx::TranslateDataType(input->GetDataType());
  auto type_size = vsi_nn_GetTypeBytes(vx_type);
  uint32_t out_size = 1;
  for (const auto& s : input->GetShape()) out_size *= s;
  out_size *= type_size;
  out_data.resize(out_size);
  if (!input->GetDataRef()) {
    return false;
  }

  std::vector<uint32_t> perm = KOcHWIc2OcIcHW;
  vx::ShapeType reverse_shape;
  for (int32_t i = input->GetShape().size() - 1; i >= 0; i--) {
    reverse_shape.push_back(input->GetShape()[i]);
  }

  vsi_nn_Transpose(out_data.data(), (uint8_t*)(input->GetDataRef()),
                   (uint32_t*)(reverse_shape.data()),
                   static_cast<uint32_t>(input->GetShape().size()),
                   perm.data(), vx_type);
  return true;
}

std::shared_ptr<vx::Tensor> OpLayoutInfer::PermuteConstTensor(
    const std::shared_ptr<vx::Tensor>& input,
    const std::shared_ptr<IPermuteVector>& pv) {
  std::vector<uint8_t> data;
  bool is_ok = TransposeConstTensorData(input, pv, data);
  assert(is_ok);
  auto src_shape = input->GetShape();
  auto dst_spec = input->GetSpec();
  vx::ShapeType dst_shape;
  for (uint32_t i = 0; i < src_shape.size(); i++) {
    dst_shape.push_back(src_shape[pv->AsStdVec()[i]]);
  }
  dst_spec.SetShape(dst_shape);
  if (dst_spec.quantization_.Type() == vx::QuantType::SYMMETRIC_PER_CHANNEL) {
    dst_spec.quantization_.SetChannelDim(
        MapAxis(pv->AsStdVec(), dst_spec.quantization_.ChannelDim()));
  }
  return context_->infer_graph_->CreateTensor(dst_spec, data.data());
}
}  // namespace transform
}  // namespace tim