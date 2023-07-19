/****************************************************************************
 *
 *    Copyright (c) 2020-2023 Vivante Corporation
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
#include "permute_vector.h"
#include "builtin_op_impl.h"
#include "tim/vx/ops/transpose.h"
#include "type_utils.h"

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
      context_->UpdateGraphOutputMap(out, context_->GetMapedTensor(out));
      auto pv = context_->GetPermuteVector(out);
      if (!pv->IsAligned()) {
        auto perm_out = InsertPermute(context_->GetMapedTensor(out),
                                      pv->Reverse(), true, out);
        // Update graph out tensor
        context_->UpdateTensorMap(out, perm_out);
        context_->UpdateGraphOutputMap(out, perm_out);
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
  auto perm_op = context_->infer_graph_->CreateOperation<vx::ops::Transpose>(
      perm->AsStdVec());
  (*perm_op).BindInput(input).BindOutput(out_tensor);
  return out_tensor;
}

std::vector<std::shared_ptr<vx::Tensor>> OpLayoutInfer::CreateOutputsTensor(
    std::shared_ptr<IPermuteVector> required_pv) {
  std::vector<std::shared_ptr<vx::Tensor>> outputs_tensor;

  if (op_->impl()->OutputsTensor().size() > 1) {
    // todo(sven): potential bug here if node have multi-output and require layout inference
    std::cout << "warning at " << __FUNCTION__ << ", #" << __LINE__
              << std::endl;
  }

  for (const auto& o : op_->impl()->OutputsTensor()) {
    auto in_shape = o->GetShape();
    auto out_spec = o->GetSpec();
    if (!(required_pv->IsAligned())) {
      out_spec = out_spec.AsTransientSpec();
    }
    auto t_infer = context_->infer_graph_->CreateTensor(out_spec);
    context_->UpdateTensorMap(o, t_infer);
    outputs_tensor.push_back(t_infer);
  }
  return outputs_tensor;
}

std::vector<std::shared_ptr<vx::Tensor>> OpLayoutInfer::CreateOutputsTensor(
    const std::vector<std::shared_ptr<IPermuteVector>>& required_pv) {
  std::vector<std::shared_ptr<vx::Tensor>> outputs_tensor;

  assert(required_pv.size() == (op_->impl()->OutputsTensor().size()));

  uint32_t i = 0;
  for (const auto& o : op_->impl()->OutputsTensor()) {
    auto in_shape = o->GetShape();
    auto out_spec = o->GetSpec();
    if (!(required_pv[i]->IsAligned())) {
      out_spec = out_spec.AsTransientSpec();
    }
    auto t_infer = context_->infer_graph_->CreateTensor(out_spec);
    context_->UpdateTensorMap(o, t_infer);
    outputs_tensor.push_back(t_infer);
    i++;
  }
  return outputs_tensor;
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
  std::shared_ptr<IPermuteVector> required_pv = nullptr;
  for (const auto& in : src_inputs) {
    if (!in->IsConstTensor()) {
      required_pv = context_->GetPermuteVector(in);
      break;
    }
  }

  if (!required_pv) {
    // all inputs are constant tensors
    for (const auto& i_src : src_inputs) {
      std::vector<uint8_t> dataRef(i_src->GetSpec().GetByteSize());
      i_src->CopyDataFromTensor(dataRef.data());
      context_->UpdateTensorMap(
          i_src, context_->infer_graph_->CreateTensor(i_src->GetSpec(),
                                                      (const void*)dataRef.data()));
      context_->SetPermuteVector(i_src, MakeShared(i_src->GetShape().size()));
    }
  } else {
    for (const auto& i_src : src_inputs) {
      std::shared_ptr<vx::Tensor> perm_out;
      if (i_src->IsConstTensor()) {
        std::vector<uint8_t> dataRef(i_src->GetSpec().GetByteSize());
        i_src->CopyDataFromTensor(dataRef.data());
        required_pv->IsAligned()
            ? perm_out = context_->infer_graph_->CreateTensor(
                  i_src->GetSpec(), (const void*)dataRef.data())
            : perm_out = PermuteConstTensor(i_src, required_pv);
      } else {
        auto final_pv =
            context_->GetPermuteVector(i_src)->Reverse()->Add(required_pv);
        final_pv->IsAligned() ? perm_out = context_->GetMapedTensor(i_src)
                              : perm_out = InsertPermute(
                                    context_->GetMapedTensor(i_src), final_pv);
      }
      context_->UpdateTensorMap(i_src, perm_out);
      context_->SetPermuteVector(i_src, required_pv);
    }
  }
  return required_pv;
}

std::shared_ptr<IPermuteVector>
OpLayoutInfer::AlignPermuteVectorForElementWise() {
  auto src_inputs = op_->impl()->InputsTensor();
  std::shared_ptr<IPermuteVector> required_pv = nullptr;
  std::shared_ptr<vx::Tensor> ref_input;
  for (const auto& in : src_inputs) {
    if (!in->IsConstTensor()) {
      required_pv = context_->GetPermuteVector(in);
      ref_input = in;
      break;
    }
  }

  for (auto i_src : src_inputs) {
    std::shared_ptr<vx::Tensor> perm_out;
    if (i_src->IsConstTensor()) {
      if (required_pv->IsAligned()) {
        std::vector<uint8_t> dataRef(i_src->GetSpec().GetByteSize());
        i_src->CopyDataFromTensor(dataRef.data());
        perm_out = context_->infer_graph_->CreateTensor(i_src->GetSpec(),
                                                        (const void*)dataRef.data());
      } else if (i_src->GetShape().size() == required_pv->Rank()) {
        perm_out = PermuteConstTensor(i_src, required_pv);
        // need shape expansion
      } else {
        auto ref_shape = ref_input->GetShape();
        auto origin_shape = i_src->GetShape();
        auto expanded_shape = GetExpandedShape(ref_shape, origin_shape);
        i_src->GetSpec().SetShape(expanded_shape);
        perm_out = PermuteConstTensor(i_src, required_pv);
      }
    } else {
      auto final_pv =
          context_->GetPermuteVector(i_src)->Reverse()->Add(required_pv);
      final_pv->IsAligned()
          ? perm_out = context_->GetMapedTensor(i_src)
          : perm_out = InsertPermute(context_->GetMapedTensor(i_src), final_pv);
    }
    context_->UpdateTensorMap(i_src, perm_out);
    context_->SetPermuteVector(i_src, required_pv);
  }
  return required_pv;
}

void OpLayoutInfer::ReverseInputsPermuteVector() {
  for (const auto& i_src : op_->impl()->InputsTensor()) {
    std::shared_ptr<vx::Tensor> perm_out;
    std::shared_ptr<IPermuteVector> input_pv;
    if (i_src->GetId() != (uint32_t)-1) {
      if (i_src->IsConstTensor()) {
        std::vector<uint8_t> dataRef(i_src->GetSpec().GetByteSize());
        i_src->CopyDataFromTensor(dataRef.data());
        perm_out = context_->infer_graph_->CreateTensor(i_src->GetSpec(),
                                                        (const void*)dataRef.data());
        input_pv = MakeShared(i_src->GetShape().size());
      } else {
        perm_out = context_->GetMapedTensor(i_src);
        input_pv = context_->GetPermuteVector(i_src);
        if (!input_pv->IsAligned()) {
          perm_out = InsertPermute(perm_out, input_pv->Reverse());
        }
      }
      context_->UpdateTensorMap(i_src, perm_out);
      context_->SetPermuteVector(i_src, MakeShared(input_pv->Rank()));
    }
  }
}

std::vector<uint32_t> OpLayoutInfer::GetExpandedShape(
    const std::vector<uint32_t>& ref_shape,
    const std::vector<uint32_t>& origin_shape) {
  std::vector<uint32_t> expanded_shape;
  for (uint32_t i = 0, j = 0; i < ref_shape.size(); ++i) {
    if (ref_shape[i] == origin_shape[j] && j < origin_shape.size()) {
      expanded_shape.push_back(origin_shape[j]);
      ++j;
    } else {
      expanded_shape.push_back(1);
    }
  }
  return expanded_shape;
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
  if (!input->IsConstTensor()) {
    return false;
  }

  vx::ShapeType reverse_shape;
  for (int32_t i = input->GetShape().size() - 1; i >= 0; i--) {
    reverse_shape.push_back(input->GetShape()[i]);
  }
  std::vector<uint32_t> perm = KOcHWIc2OcIcHW;
  std::vector<uint32_t> tmp_vec0 = kOcIcWH2WHIcOc;
  std::vector<uint32_t> tmp_vec1 = kIcOcWH2WHIcOc;
  std::vector<uint32_t> tmp_vec2 = kOcIcWHD2WHDIcOc;
  if (pv->AsStdVec() == tmp_vec0) {
    perm = kHWIcOc2OcIcHW;
  } else if (pv->AsStdVec() == tmp_vec1) {
    perm = kHWOcIc2OcIcHW;
  } else if (pv->AsStdVec() == tmp_vec2) {
    perm = kDHWIcOc2OcIcDHW;
  }

  std::vector<vsi_size_t> native_shape_array;
  std::vector<vsi_size_t> native_perm;
  std::transform(reverse_shape.begin(), reverse_shape.end(),
                 std::back_inserter(native_shape_array),
                 [](const uint32_t& i) { return i; });
  std::transform(perm.begin(), perm.end(), std::back_inserter(native_perm),
                 [](const uint32_t& i) { return i; });
  std::vector<uint8_t> dataRef(input->GetSpec().GetByteSize());
  input->CopyDataFromTensor(dataRef.data());
  vsi_nn_Transpose(out_data.data(), (uint8_t*)(dataRef.data()),
                   native_shape_array.data(),
                   static_cast<uint32_t>(input->GetShape().size()),
                   native_perm.data(), vx_type);

  return true;
}

std::shared_ptr<vx::Tensor> OpLayoutInfer::PermuteConstTensor(
    const std::shared_ptr<vx::Tensor>& input,
    const std::shared_ptr<IPermuteVector>& pv) {
  std::vector<uint8_t> data;
  bool is_ok = TransposeConstTensorData(input, pv, data);
  if (!is_ok) {
    assert(is_ok);
    return nullptr;
  }
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

std::vector<uint32_t> OpLayoutInfer::MapMultipleAxis(
    const std::vector<uint32_t>& perm, const std::vector<uint32_t>& axises) {
  assert(perm.size() == axises.size());
  std::vector<uint32_t> r(axises.size());

  for (uint32_t i = 0; i < axises.size(); ++i) {
    r[i] = axises[perm[i]];
  }

  return r;
}

std::vector<int32_t> OpLayoutInfer::MapMultipleAxis(
    const std::vector<uint32_t>& perm, const std::vector<int32_t>& axises) {
  assert(perm.size() == axises.size());
  std::vector<int32_t> r(axises.size());

  for (uint32_t i = 0; i < axises.size(); ++i) {
    r[i] = axises[perm[i]];
  }

  return r;
}

int32_t OpLayoutInfer::MapMask(const std::vector<uint32_t>& perm,
                               int32_t mask) {
  int32_t m = 0;
  for (uint32_t i = 0; i < perm.size(); ++i)
    if (mask & 1 << perm[i]) m |= (0x01 << i);
  return m;
}

}  // namespace transform
}  // namespace tim
