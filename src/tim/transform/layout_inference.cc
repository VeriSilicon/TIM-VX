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

#include "permute_vector.h"
#include "layout_infer_context.h"

#include "tim/transform/layout_inference.h"
#include "ops/conv2d_layout_inference.h"
#include "ops/grouped_conv2d_layout_inference.h"
#include "ops/reduce_layout_inference.h"
#include "ops/elementwise_layout_inference.h"
#include "ops/activation_layout_inference.h"
#include "ops/concat_layout_inferene.h"
#include "ops/simple_ops_layout_inference.h"
#include "ops/pool2d_layout_inference.h"
#include "ops/softmax_layout_inference.h"
#include "ops/squeeze_layout_inference.h"
#include "ops/stack_layout_inference.h"
#include "ops/space2depth_layout_inference.h"
#include "ops/depth2space_layout_inference.h"
#include "ops/space2batch_layout_inference.h"
#include "ops/batch2space_layout_inference.h"
#include "ops/pad_layout_inference.h"
#include "ops/pad_v2_layout_inference.h"
#include "ops/reduce_layout_inference.h"
#include "ops/fullyconnected_layout_inference.h"
#include "ops/resize_layout_inference.h"
#include "ops/split_layout_inference.h"
#include "ops/stridedslice_layout_inference.h"
#include "ops/lrn_layout_inference.h"
#include "ops/l2normalization_layout_inference.h"
#include "ops/instance_norm_layout_inference.h"
#include "ops/addn_layout_inference.h"
#include "ops/gather_layout_inference.h"
#include "ops/gather_nd_layout_inference.h"
#include "ops/reverse_layout_inference.h"
#include "ops/slice_layout_inference.h"
#include "ops/select_layout_inference.h"
#include "ops/logical_layout_inference.h"
#include "ops/arg_layout_inference.h"
#include "ops/deconv2d_layout_inference.h"
#include "ops/batchnorm_layout_inference.h"
#include "ops/conv3d_layout_inference.h"
#include "ops/default_layout_inference.h"
#include "ops/transpose_layout_inference.h"
#include "ops/yolov4_layout_inference.h"
#include "ops/unidirectional_lstm_layout_inference.h"
#include "ops/broadcast_layout_inference.h"
#include "ops/unidirectional_rnn_layout_inference.h"
#include "ops/bidirectional_rnn_layout_inference.h"
#include "ops/roi_align_layout_inference.h"
#include "ops/roi_pool_layout_inference.h"

#include <algorithm>
#include <queue>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"

namespace tim {
namespace transform {
namespace layout_inference_impl {

std::vector<std::shared_ptr<vx::Tensor>> HandleLayoutInfer(
    std::shared_ptr<layout_inference_impl::LayoutInferContext>& ctx,
    const std::shared_ptr<vx::Operation>& op);

// Implementation for LayoutInferContext
LayoutInferContext::LayoutInferContext(
    const std::shared_ptr<vx::Graph>& src_graph,
    std::shared_ptr<vx::Graph>& infer_graph)
    : src_graph_(src_graph), infer_graph_(infer_graph) {
  for (const auto& op : src_graph->OpVector()) {
    op_visited_[op] = false;
  }
}

void LayoutInferContext::SetPermuteVector(std::shared_ptr<vx::Tensor> tensor,
                                          std::shared_ptr<IPermuteVector> pv) {
  if (tensor_pv_.end() != tensor_pv_.find(tensor)) {
    VSILOGD("Tensor PermuteVector has been setted.");
  }
  tensor_pv_[tensor] = pv;
}

const std::shared_ptr<IPermuteVector> LayoutInferContext::GetPermuteVector(
    const std::shared_ptr<vx::Tensor>& tensor) const {
  auto pv_it = tensor_pv_.find(tensor);
  if (pv_it != tensor_pv_.end()) {
    return pv_it->second;
  } else {
    VSILOGE("Tensor PermuteVecor has not beed setted.");
    assert(false);
  }

  return nullptr;
}

void LayoutInferContext::MarkVisited(const std::shared_ptr<vx::Operation>& op) {
  op_visited_[op] = true;
}

bool LayoutInferContext::IsVisited(
    const std::shared_ptr<vx::Operation>& op) const {
  return op_visited_.at(op);
}

bool LayoutInferContext::IsReadyForInfer(
    const std::shared_ptr<vx::Operation>& op) const {
  for (const auto& tensor : op->impl()->InputsTensor()) {
    if (!tensor->IsConstTensor() &&
        tensor->GetId() != static_cast<uint32_t>(-1) &&
        (tensor_pv_.end() == tensor_pv_.find(tensor))) {
      return false;
    }
  }
  return true;
}

void LayoutInferContext::UpdateTensorMap(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::shared_ptr<vx::Tensor>& t_layout) {
  tensor_map_[t_src] = t_layout;
}

std::shared_ptr<vx::Tensor> LayoutInferContext::GetMappedTensor(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = tensor_map_.find(t_src);
  if (it != tensor_map_.end()) {
    return it->second;
  }

  VSILOGE("Tensor has not beed inserted in tensor map.");
  return nullptr;
}

std::shared_ptr<vx::Tensor> LayoutInferContext::GetMappedGraphInputTensor(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = graph_input_map_.find(t_src);
  if (it != tensor_map_.end()) {
    return it->second;
  }

  VSILOGE("Tensor has not beed inserted in graph input tensor map.");
  return nullptr;
}

std::shared_ptr<vx::Tensor> LayoutInferContext::GetMappedGraphOutputTensor(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = graph_output_map_.find(t_src);
  if (it != tensor_map_.end()) {
    return it->second;
  }

  VSILOGE("Tensor has not beed inserted in graph output tensor map.");
  return nullptr;
}

void LayoutInferContext::UpdateGraphInputMap(
    const std::shared_ptr<vx::Tensor>& i_src,
    const std::shared_ptr<vx::Tensor>& i_layout) {
  graph_input_map_[i_src] = i_layout;
}

void LayoutInferContext::UpdateGraphOutputMap(
    const std::shared_ptr<vx::Tensor>& o_src,
    const std::shared_ptr<vx::Tensor>& o_layout) {
  graph_output_map_[o_src] = o_layout;
}

#define REGISTER_LAYOUT_INFERENCE(op_idx, name)                   \
  case op_idx: {                                                  \
    auto op_infer = std::make_shared<name##LayoutInfer>(op, ctx); \
    op_infer->OnInputs(next_tensors);                             \
    op_infer->OnOutputs(next_tensors);                            \
    break;                                                        \
  }

#define REGISTER_REDUCE_LAYOUT_INFERENCE(op_idx)                    \
  case op_idx: {                                                    \
    auto reduce_type = op->impl()->node()->nn_param.reduce.type;    \
    switch (reduce_type) {                                          \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_MEAN, ReduceMean);    \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_MAX, ReduceMax);      \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_MIN, ReduceMin);      \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_PROD, ReduceProd);    \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_ANY, ReduceAny);      \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_SUM, ReduceSum);      \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_REDUCE_ALL, ReduceAll);      \
      default:                                                      \
        VSILOGW("Op %d: Default layout inference pass for reduce.", \
                reduce_type);                                       \
        assert(false);                                              \
    }                                                               \
    break;                                                          \
  }

#define REGISTER_LOGICAL_LAYOUT_INFERENCE(op_idx)                       \
  case op_idx: {                                                        \
    auto logical_type = op->impl()->node()->nn_param.relational_ops.op; \
    switch (logical_type) {                                             \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_LOGICAL_AND, LogicalAnd);        \
      REGISTER_LAYOUT_INFERENCE(VSI_NN_LOGICAL_OR, LogicalOr);          \
      default:                                                          \
        VSILOGW("Op %d: Default layout inference pass for logical.",    \
                logical_type);                                          \
        assert(false);                                                  \
    }                                                                   \
    break;                                                              \
  }

std::vector<std::shared_ptr<vx::Tensor>> HandleLayoutInfer(
    std::shared_ptr<layout_inference_impl::LayoutInferContext>& ctx,
    const std::shared_ptr<vx::Operation>& op) {
  ctx->MarkVisited(op);
  auto op_id = op->impl()->kind_;
  std::vector<std::shared_ptr<vx::Tensor>> next_tensors;
  switch (op_id) {
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_CONV2D, Conv2d);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_GROUPED_CONV2D, GroupedConv2d);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_RELU, Relu);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_RELU1, Relu1);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_RELU6, Relu6);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ELU, Elu);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SIGMOID, Sigmoid);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_MISH, Mish);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_HARD_SIGMOID, HardSigmoid);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SOFTRELU, SoftRelu);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SWISH, HardSwish);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_LEAKY_RELU, LeakyRelu);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_CONCAT, Concat);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ADD, Add);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SUBTRACT, Sub);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_MULTIPLY, Multiply);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_DIVIDE, Div);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_POW, Pow);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_MINIMUM, Minimum);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_MAXIMUM, Maximum);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_DATACONVERT, DataConvert);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_NEG, Neg);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ABS, Abs);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SIN, Sin);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_TANH, Tanh);
#ifdef VSI_FEAT_OP_COS
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_COS, Cos);
#endif
#ifdef VSI_FEAT_OP_TAN
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_TAN, Tan);
#endif
#ifdef VSI_FEAT_OP_ATAN
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ATAN, ATan);
#endif
#ifdef VSI_FEAT_OP_ATANH
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ATANH, ATanh);
#endif
#ifdef VSI_FEAT_OP_ACOSH
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ACOSH, ACosh);
#endif
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_EXP, Exp);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_LOG, Log);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SQRT, Sqrt);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_RSQRT, Rsqrt);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SQUARE, Square);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_LOGICAL_NOT, LogicalNot);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_POOL, Pool2d);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SOFTMAX, Softmax);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SQUEEZE, Squeeze);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_STACK, Stack);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SPACE2DEPTH, SpaceToDepth);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_DEPTH2SPACE, DepthToSpace);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SPACE2BATCH, Space2Batch);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_BATCH2SPACE, Batch2Space);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_PAD, Pad);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_PAD2, PadV2);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_FCL2, FullyConnected);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_RESIZE, Resize);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SPLIT, Split);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_STRIDED_SLICE, StridedSlice);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_LRN2, LRN);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_L2_NORMALIZE, L2Normalization);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_INSTANCE_NORM, InstanceNorm);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ROI_ALIGN, RoiAlign);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ROI_POOL, RoiPool);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ADDN, AddN);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_PRELU, PRelu);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_GATHER, Gather);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_GATHER_ND, GatherNd);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_REVERSE, Reverse);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SLICE, Slice);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_SELECT, Select);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ARGMAX, Arg);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_ARGMIN, Arg);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_DECONVOLUTION, DeConv2d);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_BATCH_NORM, BatchNorm);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_PERMUTE, Transpose);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_CONV3D, Conv3d);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_LSTM_OVXLIB, UnidirectionalLstm);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_EXPAND_BROADCAST, Broadcast);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_UNIDIRECTIONAL_SEQUENCE_RNN,
                              UnidirectionalRnn);
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_RNN,
                              BidirectionalRnn);
#ifdef VSI_FEAT_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS
    REGISTER_LAYOUT_INFERENCE(VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS, Yolov4);
#endif
    REGISTER_LOGICAL_LAYOUT_INFERENCE(VSI_NN_OP_LOGICAL_OPS);
    REGISTER_REDUCE_LAYOUT_INFERENCE(VSI_NN_OP_REDUCE);
    // use default layout inference
    default: {
      VSILOGW("Op %d: default layout inference pass.", op_id);
      auto op_infer = std::make_shared<DefaultLayoutInfer>(op, ctx);
      op_infer->OnInputs(next_tensors);
      op_infer->OnOutputs(next_tensors);
    }
  }
  return next_tensors;
}
}  // namespace layout_inference_impl

std::pair<std::shared_ptr<vx::Graph>,
          std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>>
LayoutInference(
    const std::shared_ptr<vx::Graph>& src_graph,
    std::shared_ptr<vx::Context>& ctx,
    std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<IPermuteVector>>
        tensor_pv_map) {
  std::shared_ptr<vx::Graph> infer_graph = ctx->CreateGraph();
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_io_map;
  auto layout_infer_ctx =
      std::make_shared<layout_inference_impl::LayoutInferContext>(src_graph,
                                                                  infer_graph);

  std::queue<std::shared_ptr<vx::Tensor>> tensor_queue;
  auto graph_inputs = src_graph->InputsTensor();
  for (const auto& t_src : graph_inputs) {
    auto input = infer_graph->CreateTensor(t_src->GetSpec());
    layout_infer_ctx->UpdateTensorMap(t_src, input);
    layout_infer_ctx->UpdateGraphInputMap(t_src, input);
    tensor_queue.push(t_src);
    layout_infer_ctx->SetPermuteVector(
        t_src, tensor_pv_map.find(t_src) != tensor_pv_map.end()
                   ? tensor_pv_map[t_src]
                   : MakeShared(t_src->GetShape().size()));
  }

  auto const_inputs = src_graph->GetConstantInputs();
  for (auto const_in : const_inputs) {
    std::vector<uint8_t> dataRef(const_in->GetSpec().GetByteSize());
    const_in->CopyDataFromTensor(dataRef.data());
    auto input = infer_graph->CreateTensor(const_in->GetSpec(),
                                           (const void*)dataRef.data());
    layout_infer_ctx->UpdateTensorMap(const_in, input);
    tensor_queue.push(const_in);
    layout_infer_ctx->SetPermuteVector(
        const_in, tensor_pv_map.find(const_in) != tensor_pv_map.end()
                      ? tensor_pv_map[const_in]
                      : MakeShared(const_in->GetShape().size()));
  }

  auto graph_outputs = src_graph->OutputsTensor();
  for (const auto& t_src : graph_outputs) {
    auto output = infer_graph->CreateTensor(t_src->GetSpec());
    layout_infer_ctx->UpdateTensorMap(t_src, output);
    layout_infer_ctx->UpdateGraphOutputMap(t_src, output);
    tensor_queue.push(t_src);
    layout_infer_ctx->SetPermuteVector(
        t_src, tensor_pv_map.find(t_src) != tensor_pv_map.end()
                   ? tensor_pv_map[t_src]
                   : MakeShared(t_src->GetShape().size()));
  }

  while (!tensor_queue.empty()) {
    auto tensor = tensor_queue.front();
    tensor_queue.pop();
    const auto& consumers = src_graph->GetConsumersOp(tensor);
    for (const auto& op : consumers) {
      if (!layout_infer_ctx->IsVisited(op) && op->impl()->kind_ != -1 &&
          layout_infer_ctx->IsReadyForInfer(op)) {
        auto next_tensors =
            layout_inference_impl::HandleLayoutInfer(layout_infer_ctx, op);
        for (const auto& t : next_tensors) {
          tensor_queue.push(t);
        }
      }
    }
  }
  for (const auto& graph_input : layout_infer_ctx->GetGraphInputMap()) {
    graph_io_map[graph_input.first] = graph_input.second;
  }
  for (const auto& graph_output : layout_infer_ctx->GetGraphOutputMap()) {
    graph_io_map[graph_output.first] = graph_output.second;
  }
  return std::make_pair(infer_graph, graph_io_map);
}

}  // namespace transform
}  // namespace tim