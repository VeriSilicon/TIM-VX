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

#include "permute_vector.h"
#include "layout_infer_context.h"

#include "tim/transform/layout_inference.h"
#include "ops/conv2d_layout_inference.h"
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
#include "ops/reduce_layout_inference.h"
#include "ops/fullyconnected_layout_inference.h"
#include "ops/resize_layout_inference.h"
#include "ops/split_layout_inference.h"
#include "ops/stridedslice_layout_inference.h"
#include "ops/lrn_layout_inference.h"
#include "ops/l2normalization_layout_inference.h"
#include "ops/addn_layout_inference.h"
#include "ops/gather_layout_inference.h"
#include "ops/gather_nd_layout_inference.h"
#include "ops/reverse_layout_inference.h"
#include "ops/slice_layout_inference.h"
#include "ops/select_layout_inference.h"
#include "ops/logical_layout_inference.h"
#include "ops/arg_layout_inference.h"
#include "ops/deconv2d_layout_inference.h"
#include "ops/default_layout_inference.h"

#include <algorithm>
#include <deque>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"

namespace tim {
namespace transform {
namespace layout_inference_impl {

std::vector<std::shared_ptr<vx::Tensor>> HandleLayoutInfer(
    std::shared_ptr<layout_inference_impl::LayoutInferContext>& ctx,
    const std::shared_ptr<vx::Operation>& op);

// Implemention for LayoutInferContext
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
  if (visited_op_.end() !=
      std::find(visited_op_.begin(), visited_op_.end(), op)) {
    VSILOGW("The operation has been mark as visited.");
  } else {
    visited_op_.push_back(op);
  }
}

bool LayoutInferContext::IsVisited(const std::shared_ptr<vx::Operation>& op) const {
  if (visited_op_.end() !=
      std::find(visited_op_.begin(), visited_op_.end(), op)) {
    return true;
  } else {
    return false;
  }
}

bool LayoutInferContext::IsReadyForInfer(
    const std::shared_ptr<vx::Operation>& op) const {
  for (const auto& tensor : op->impl()->InputsTensor()) {
    if (!tensor->IsConstTensor() &&
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

std::shared_ptr<vx::Tensor> LayoutInferContext::GetMapedTensor(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = tensor_map_.find(t_src);
  if (it != tensor_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in tensor map.");
    assert(false);
  }

  return nullptr;
}

void LayoutInferContext::UpdateGraphInputMap(const std::shared_ptr<vx::Tensor>& i_src,
                           const std::shared_ptr<vx::Tensor>& i_layout) {
  graph_input_map_[i_src] = i_layout;
}

#define REGIST_LAYOUT_INFERENCE(op_idx, name)                     \
  case op_idx: {                                                  \
    auto op_infer = std::make_shared<name##LayoutInfer>(op, ctx); \
    op_infer->OnInputs(next_tensors);                             \
    op_infer->OnOutputs(next_tensors);                            \
    break;                                                        \
  }                                                               \

#define REGIST_REDUCE_LAYOUT_INFERENCE(op_idx)                                 \
  case op_idx: {                                                               \
    auto reduce_type = op->impl()->node()->nn_param.reduce.type;               \
    switch (reduce_type) {                                                     \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_MEAN, ReduceMean);                 \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_MAX, ReduceMax);                   \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_MIN, ReduceMin);                   \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_PROD, ReduceProd);                 \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_ANY, ReduceAny);                   \
      REGIST_LAYOUT_INFERENCE(VSI_NN_REDUCE_SUM, ReduceSum);                   \
    default:                                                                   \
      VSILOGW("Op %d: Default layout inference pass for reduce.", reduce_type);\
      assert(false);                                                           \
    }                                                                          \
    break;                                                                     \
  }                                                                            \

#define REGIST_LOGICAL_LAYOUT_INFERENCE(op_idx)                                  \
  case op_idx: {                                                                 \
    auto logical_type = op->impl()->node()->nn_param.relational_ops.op;          \
    switch (logical_type)                                                        \
    {                                                                            \
      REGIST_LAYOUT_INFERENCE(VSI_NN_LOGICAL_AND, LogicalAnd);                   \
      REGIST_LAYOUT_INFERENCE(VSI_NN_LOGICAL_OR, LogicalOr);                     \
    default:                                                                     \
      VSILOGW("Op %d: Default layout inference pass for logical.", logical_type);\
      assert(false);                                                             \
    }                                                                            \
    break;                                                                       \
  }                                                                              \

std::vector<std::shared_ptr<vx::Tensor>> HandleLayoutInfer(
    std::shared_ptr<layout_inference_impl::LayoutInferContext>& ctx,
    const std::shared_ptr<vx::Operation>& op) {
  ctx->MarkVisited(op);
  auto op_id = op->impl()->operation_id_;
  std::vector<std::shared_ptr<vx::Tensor>> next_tensors;
  switch (op_id) {
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_CONV2D, Conv2d);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_RELU, Relu);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_RELU1, Relu1);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_RELU6, Relu6);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ELU, Elu);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SIGMOID, Sigmoid);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_MISH, Mish);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_HARD_SIGMOID, HardSigmoid);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SOFTRELU, SoftRelu);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SWISH, HardSwish);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_TANH, Tanh);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_LEAKY_RELU, LeakyRelu);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_CONCAT, Concat);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ADD, Add);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SUBTRACT, Sub);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_MULTIPLY, Multiply);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_DIVIDE, Div);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_POW, Pow);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_MINIMUM, Minimum);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_MAXIMUM, Maximum);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_DATACONVERT, DataConvert);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_NEG, Neg);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ABS, Abs);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SIN, Sin);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_EXP, Exp);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_LOG, Log);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SQRT, Sqrt);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_RSQRT, Rsqrt);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SQUARE, Square);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_LOGICAL_NOT, LogicalNot);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_POOL, Pool2d);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SOFTMAX, Softmax);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SQUEEZE, Squeeze);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_STACK, Stack);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SPACE2DEPTH, SpaceToDepth);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_DEPTH2SPACE, DepthToSpace);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SPACE2BATCH, Space2Batch);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_BATCH2SPACE, Batch2Space);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_PAD, Pad);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_FCL2, FullyConnected);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_RESIZE, Resize);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SPLIT, Split);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_STRIDED_SLICE, StridedSlice);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_LRN2, LRN);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_L2_NORMALIZE, L2Normalization);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ADDN, AddN);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_PRELU, PRelu);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_GATHER, Gather);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_GATHER_ND, GatherNd);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_REVERSE, Reverse);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SLICE, Slice);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_SELECT, Select);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ARGMAX, Arg);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_ARGMIN, Arg);
    REGIST_LAYOUT_INFERENCE(VSI_NN_OP_DECONVOLUTION, DeConv2d);
    REGIST_LOGICAL_LAYOUT_INFERENCE(VSI_NN_OP_LOGICAL_OPS);
    REGIST_REDUCE_LAYOUT_INFERENCE(VSI_NN_OP_REDUCE);
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
          std::map<std::shared_ptr<vx::Tensor>,
                   std::shared_ptr<vx::Tensor>>> LayoutInference(
    const std::shared_ptr<vx::Graph>& src_graph,
    std::shared_ptr<vx::Context>& ctx) {
  std::shared_ptr<vx::Graph> infer_graph = ctx->CreateGraph();
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_io_map;
  auto layout_infer_ctx =
      std::make_shared<layout_inference_impl::LayoutInferContext>(src_graph,
                                                                  infer_graph);

  std::deque<std::shared_ptr<vx::Tensor>> tensor_queue;
  auto graph_inputs = src_graph->InputsTensor();
  for (const auto& t_src : graph_inputs) {
    auto input = infer_graph->CreateTensor(t_src->GetSpec());
    layout_infer_ctx->UpdateTensorMap(t_src, input);
    layout_infer_ctx->UpdateGraphInputMap(t_src, input);
    tensor_queue.push_back(t_src);
    layout_infer_ctx->SetPermuteVector(t_src,
                                       MakeShared(t_src->GetShape().size()));
  }

  while (!tensor_queue.empty()) {
    const auto& tensor = tensor_queue.front();
    tensor_queue.pop_front();
    const auto& consumers = src_graph->GetConsumersOp(tensor);
    for (const auto& op : consumers) {
      if (!layout_infer_ctx->IsVisited(op) &&
          layout_infer_ctx->IsReadyForInfer(op)) {
        auto next_tensors =
            layout_inference_impl::HandleLayoutInfer(layout_infer_ctx, op);
        for (const auto& t : next_tensors) {
          tensor_queue.push_back(t);
        }
      }
    }
  }
  for (const auto& graph_input : layout_infer_ctx->GetGraphInputMap()) {
    graph_io_map[graph_input.first] = graph_input.second;
  }
  for (const auto& out_src : src_graph->OutputsTensor()) {
    graph_io_map[out_src] = layout_infer_ctx->GetMapedTensor(out_src);
  }
  return std::make_pair(infer_graph, graph_io_map);
}

}  // namespace transform
}  // namespace tim