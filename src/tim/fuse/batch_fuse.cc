
#include "tim/fuse/batch_fuse.h"
// #include "builtin_op_impl.h"
#include "batch_fuse_context.h"
#include <algorithm>
#include <deque>
#include <cassert>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"

#include "ops/conv2d_batch_fuse.h"
// #include "ops/pad_v2_batch_fuse.h"
#include "ops/pad_batch_fuse.h"
#include "ops/activation_batch_fuse.h"
#include "ops/elementwise_batch_fuse.h"
#include "ops/pool2d_batch_fuse.h"
#include "ops/reduce_batch_fuse.h"
#include "ops/transpose_batch_fuse.h"
#include "ops/reshape_batch_fuse.h"
#include "ops/concat_batch_fuse.h"

namespace tim {
namespace fuse {
namespace batch_fuse_impl {

std::vector<std::shared_ptr<vx::Tensor>> HandleBatchFuse(
    std::shared_ptr<batch_fuse_impl::BatchFuseContext>& ctx,
    const std::shared_ptr<vx::Operation>& op);

void BatchFuseContext::MarkVisited(const std::shared_ptr<vx::Operation>& op) {
  if (visited_op_.end() !=
      std::find(visited_op_.begin(), visited_op_.end(), op)) {
    VSILOGW("The operation has been mark as visited.");
  } else {
    visited_op_.push_back(op);
  }
}

bool BatchFuseContext::IsVisited(
    const std::shared_ptr<vx::Operation>& op) const {
  if (visited_op_.end() !=
      std::find(visited_op_.begin(), visited_op_.end(), op)) {
    return true;
  } else {
    return false;
  }
}

bool BatchFuseContext::IsReadyForBatchFuse(
    const std::shared_ptr<vx::Operation>& op) const {
  for (const auto& tensor : op->impl()->InputsTensor()) {
    if (!tensor->IsConstTensor() && tensor->GetId() != (uint32_t)-1 &&
        (tensor_map_.end() == tensor_map_.find(tensor))) {
      return false;
    }
  }
  return true;
}

bool BatchFuseContext::IsReadyForPadInfer(
    const std::shared_ptr<vx::Operation>& op) const {
  for (const auto& tensor : op->impl()->InputsTensor()) {
    if (!tensor->IsConstTensor() && tensor->GetId() != (uint32_t)-1 &&
        (pad_infer_shape_map_.end() == pad_infer_shape_map_.find(tensor))) {
      return false;
    }
  }
  return true;
}

bool BatchFuseContext::IsReadyForCloneGraph(
    const std::shared_ptr<vx::Operation>& op) const {
  for (const auto& tensor : op->impl()->InputsTensor()) {
    if (!tensor->IsConstTensor() && tensor->GetId() != (uint32_t)-1 &&
        (clone_tensor_map_.end() == clone_tensor_map_.find(tensor))) {
      return false;
    }
  }
  return true;
}

void BatchFuseContext::UpdateTensorBatchFuseMap(
    const std::shared_ptr<vx::Tensor>& t_batch_fuse,
    const std::shared_ptr<vx::Tensor>& t_src) {
  tensor_batch_fuse_map_[t_batch_fuse] = t_src;
}

void BatchFuseContext::UpdateTensorMap(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::shared_ptr<vx::Tensor>& t_batch_fuse) {
  tensor_map_[t_src] = t_batch_fuse;
}

void BatchFuseContext::UpdateCloneTensorMap(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::shared_ptr<vx::Tensor>& t_clone_graph) {
  clone_tensor_map_[t_src] = t_clone_graph;
}

void BatchFuseContext::UpdateInitPad(const std::shared_ptr<vx::Tensor>& t_src,
                                     const std::array<uint32_t, 4>& pad) {
  init_pad_map_[t_src] = pad;
}

void BatchFuseContext::UpdateBackwardPad(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::array<uint32_t, 4>& pad) {
  backward_pad_map_[t_src] = pad;
}

void BatchFuseContext::UpdateForwardPad(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::array<uint32_t, 4>& pad) {
  forward_pad_map_[t_src] = pad;
}

void BatchFuseContext::UpdateBackwardGap(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::array<uint32_t, 2>& pad) {
  backward_gap_map_[t_src] = pad;
}

void BatchFuseContext::UpdateForwardGap(
    const std::shared_ptr<vx::Tensor>& t_src,
    const std::array<uint32_t, 2>& pad) {
  forward_gap_map_[t_src] = pad;
}

void BatchFuseContext::UpdatePadInferShape(
    const std::shared_ptr<vx::Tensor>& t_src, const ShapeType& shape) {
  pad_infer_shape_map_[t_src] = shape;
}

void BatchFuseContext::UpdateProportion(
    const std::shared_ptr<vx::Tensor>& t_src, int32_t pro) {
  proportion_map_[t_src] = pro;
}

std::shared_ptr<vx::Tensor> BatchFuseContext::GetMapedTensor(
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

std::shared_ptr<vx::Tensor> BatchFuseContext::GetCloneMapedTensor(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = clone_tensor_map_.find(t_src);
  if (it != clone_tensor_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in tensor map.");
    assert(false);
  }

  return nullptr;
}

std::shared_ptr<vx::Tensor> BatchFuseContext::GetBatchFuseMapedTensor(
    const std::shared_ptr<vx::Tensor>& t_batch_fuse) const {
  auto it = tensor_map_.find(t_batch_fuse);
  if (it != tensor_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in batch fuse tensor map.");
    assert(false);
  }

  return nullptr;
}

std::array<uint32_t, 4> BatchFuseContext::GetInitPad(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = init_pad_map_.find(t_src);
  if (it != init_pad_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in init pad map.");
    assert(false);
  }

  return {};
}

std::array<uint32_t, 4> BatchFuseContext::GetBackwardPad(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = backward_pad_map_.find(t_src);
  if (it != backward_pad_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in backward pad map.");
    assert(false);
  }

  return {};
}

std::array<uint32_t, 2> BatchFuseContext::GetBackwardGap(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = backward_gap_map_.find(t_src);
  if (it != backward_gap_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in backward gap map.");
    assert(false);
  }

  return {};
}

std::array<uint32_t, 4> BatchFuseContext::GetForwardPad(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = forward_pad_map_.find(t_src);
  if (it != forward_pad_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in forward pad map.");
    assert(false);
  }

  return {};
}

std::array<uint32_t, 2> BatchFuseContext::GetForwardGap(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = forward_gap_map_.find(t_src);
  if (it != forward_gap_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in forward gap map.");
    assert(false);
  }

  return {};
}

ShapeType BatchFuseContext::GetPadInferShape(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = pad_infer_shape_map_.find(t_src);
  if (it != pad_infer_shape_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in pad infer shape map.");
    assert(false);
  }

  return {};
}

int32_t BatchFuseContext::GetProportion(
    const std::shared_ptr<vx::Tensor>& t_src) const {
  auto it = proportion_map_.find(t_src);
  if (it != proportion_map_.end()) {
    return it->second;
  } else {
    VSILOGE("Tensor has not beed inserted in proportion map.");
    assert(false);
  }

  return 0;
}

void BatchFuseContext::UpdateGraphInputMap(
    const std::shared_ptr<vx::Tensor>& i_src,
    const std::shared_ptr<vx::Tensor>& i_batch_fuse) {
  graph_input_map_[i_src] = i_batch_fuse;
}

void BatchFuseContext::UpdateGraphOutputMap(
    const std::shared_ptr<vx::Tensor>& o_src,
    const std::shared_ptr<vx::Tensor>& o_batch_fuse) {
  graph_output_map_[o_src] = o_batch_fuse;
}

#define REGIST_BATCH_FUSE(op_idx, name, Backward, Forward, Clone)        \
  case op_idx: {                                                         \
    auto op_batch_fuse = std::make_shared<name##BatchFuse>(op, ctx);     \
    if (Forward) {                                                       \
      need_backward = op_batch_fuse->PadForwardInference(next_tensors);  \
    } else if (Backward) {                                               \
      need_backward = op_batch_fuse->PadBackwardInference(next_tensors); \
    } else if (Clone) {                                                  \
      op_batch_fuse->CloneGraph(next_tensors);                           \
    } else {                                                             \
      op_batch_fuse->OnInputs(next_tensors);                             \
      op_batch_fuse->OnOutputs(next_tensors);                            \
    }                                                                    \
    break;                                                               \
  }  // namespace batch_fuse_impl

#define REGIST_REDUCE_BATCH_FUSE(op_idx, Backward, Forward, Clone)         \
  case op_idx: {                                                           \
    auto reduce_type = op->impl()->node()->nn_param.reduce.type;           \
    switch (reduce_type) {                                                 \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_MEAN, ReduceMean, Backward, Forward, \
                        Clone);                                            \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_MAX, ReduceMax, Backward, Forward,   \
                        Clone);                                            \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_MIN, ReduceMin, Backward, Forward,   \
                        Clone);                                            \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_PROD, ReduceProd, Backward, Forward, \
                        Clone);                                            \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_ANY, ReduceAny, Backward, Forward,   \
                        Clone);                                            \
      REGIST_BATCH_FUSE(VSI_NN_REDUCE_SUM, ReduceSum, Backward, Forward,   \
                        Clone);                                            \
      default:                                                             \
        VSILOGW("Op %d: Default layout inference pass for reduce.",        \
                reduce_type);                                              \
        assert(false);                                                     \
    }                                                                      \
    break;                                                                 \
  }

std::vector<std::shared_ptr<vx::Tensor>> HandleBatchFuse(
    std::shared_ptr<batch_fuse_impl::BatchFuseContext>& ctx,
    const std::shared_ptr<vx::Operation>& op, bool Backward, bool Forward,
    bool Clone) {
  ctx->MarkVisited(op);
  auto op_id = op->impl()->kind_;
  bool need_backward = false;
  std::vector<std::shared_ptr<vx::Tensor>> next_tensors;
  switch (op_id) {
    REGIST_BATCH_FUSE(VSI_NN_OP_CONV2D, Conv2d, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PAD, Pad, Backward, Forward, Clone);
    // REGIST_BATCH_FUSE(VSI_NN_OP_PAD2, PadV2, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RELU, Relu, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_ADD, Add, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_POOL, Pool2d, Backward, Forward, Clone);
    REGIST_REDUCE_BATCH_FUSE(VSI_NN_OP_REDUCE, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PERMUTE, Transpose, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RESHAPE2, Reshape, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_CONCAT, Concat, Backward, Forward, Clone);
  }
  return next_tensors;
}

std::pair<std::vector<std::shared_ptr<vx::Tensor>>, bool> HandlePadInfernce(
    std::shared_ptr<batch_fuse_impl::BatchFuseContext>& ctx,
    const std::shared_ptr<vx::Operation>& op, bool Backward, bool Forward,
    bool Clone) {
  auto op_id = op->impl()->kind_;
  std::vector<std::shared_ptr<vx::Tensor>> next_tensors;
  bool need_backward = false;
  switch (op_id) {
    REGIST_BATCH_FUSE(VSI_NN_OP_CONV2D, Conv2d, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PAD, Pad, Backward, Forward, Clone);
    // REGIST_BATCH_FUSE(VSI_NN_OP_PAD2, PadV2, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RELU, Relu, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_ADD, Add, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_POOL, Pool2d, Backward, Forward, Clone);
    REGIST_REDUCE_BATCH_FUSE(VSI_NN_OP_REDUCE, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PERMUTE, Transpose, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RESHAPE, Reshape, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RESHAPE2, Reshape, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_CONCAT, Concat, Backward, Forward, Clone);
  }
  return std::make_pair(next_tensors, need_backward);
}

std::vector<std::shared_ptr<vx::Tensor>> HandleCloneGraph(
    std::shared_ptr<batch_fuse_impl::BatchFuseContext>& ctx,
    const std::shared_ptr<vx::Operation>& op, bool Backward, bool Forward,
    bool Clone) {
  auto op_id = op->impl()->kind_;
  std::vector<std::shared_ptr<vx::Tensor>> next_tensors;
  bool need_backward = false;
  switch (op_id) {
    REGIST_BATCH_FUSE(VSI_NN_OP_CONV2D, Conv2d, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PAD, Pad, Backward, Forward, Clone);
    // REGIST_BATCH_FUSE(VSI_NN_OP_PAD2, PadV2, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RELU, Relu, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_ADD, Add, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_POOL, Pool2d, Backward, Forward, Clone);
    REGIST_REDUCE_BATCH_FUSE(VSI_NN_OP_REDUCE, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_PERMUTE, Transpose, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RESHAPE, Reshape, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_RESHAPE2, Reshape, Backward, Forward, Clone);
    REGIST_BATCH_FUSE(VSI_NN_OP_CONCAT, Concat, Backward, Forward, Clone);
  }
  return next_tensors;
}

}  // namespace batch_fuse_impl

std::pair<
    /*graph after batch fused*/
    std::shared_ptr<vx::Graph>,
    /* tensor mapping between original graph and graph after batch fused*/
    std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>>
BatchFuse(const std::shared_ptr<vx::Graph>& src_graph,
          std::shared_ptr<vx::Context>& ctx, uint32_t fake_batch) {
  std::shared_ptr<vx::Graph> batch_fuse_graph = ctx->CreateGraph();
  std::shared_ptr<vx::Graph> clone_batch_graph = ctx->CreateGraph();
  auto batch_fuse_ctx = std::make_shared<batch_fuse_impl::BatchFuseContext>(
      src_graph, batch_fuse_graph, clone_batch_graph);
  batch_fuse_ctx->SetFakeBatch(fake_batch);

  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_io_map;
  std::deque<std::shared_ptr<vx::Tensor>> tensor_queue;
  std::deque<std::shared_ptr<vx::Tensor>> tensor_queue_pad_forward;
  std::deque<std::shared_ptr<vx::Tensor>> tensor_queue_pad_backward;
  std::deque<std::shared_ptr<vx::Tensor>> tensor_clone_queue;

  auto graph_inputs = src_graph->InputsTensor();
  for (const auto& t_src : graph_inputs) {
    auto input = clone_batch_graph->CreateTensor(t_src->GetSpec());
    batch_fuse_ctx->UpdateCloneTensorMap(t_src, input);
    batch_fuse_ctx->UpdateGraphInputMap(t_src, input);
    tensor_clone_queue.push_back(t_src);
  }

  auto const_inputs = src_graph->GetConstantInputs();
  for (auto const_in : const_inputs) {
    std::vector<uint8_t> tmp(const_in->GetSpec().GetByteSize());
    const_in->CopyDataFromTensor(tmp.data());
    auto input = clone_batch_graph->CreateTensor(const_in->GetSpec(),
                                                 tmp.data());
    batch_fuse_ctx->UpdateCloneTensorMap(const_in, input);
    tensor_clone_queue.push_back(const_in);
  }

  //clone graph to multi batch
  while (!tensor_clone_queue.empty()) {
    auto tensor = tensor_clone_queue.front();
    tensor_clone_queue.pop_front();
    const auto& consumers = src_graph->GetConsumersOp(tensor);
    for (const auto& op : consumers) {
      if (op->impl()->kind_ != -1 && batch_fuse_ctx->IsReadyForCloneGraph(op)) {
        auto next_tensors = batch_fuse_impl::HandleCloneGraph(
            batch_fuse_ctx, op, false, false, true);
        for (const auto& t : next_tensors) {
          tensor_clone_queue.push_back(t);
        }
      }
    }
  }

  auto graph_inputs_clone = clone_batch_graph->InputsTensor();
  for (const auto& t_src : graph_inputs_clone) {
    auto input = batch_fuse_graph->CreateTensor(t_src->GetSpec());
    batch_fuse_ctx->UpdateTensorMap(t_src, input);
    batch_fuse_ctx->UpdateTensorBatchFuseMap(input, t_src);
    batch_fuse_ctx->UpdateGraphInputMap(t_src, input);
    batch_fuse_ctx->UpdatePadInferShape(t_src, t_src->GetShape());
    batch_fuse_ctx->UpdateForwardPad(t_src, {0, 0, 0, 0});
    batch_fuse_ctx->UpdateForwardGap(t_src, {0, 0});
    tensor_queue.push_back(t_src);
    tensor_queue_pad_forward.push_back(t_src);
    tensor_clone_queue.push_back(t_src);
  }

  auto const_inputs_clone = clone_batch_graph->GetConstantInputs();
  for (auto const_in : const_inputs_clone) {
    std::vector<uint8_t> tmp(const_in->GetSpec().GetByteSize());
    const_in->CopyDataFromTensor(tmp.data());
    auto input = batch_fuse_graph->CreateTensor(const_in->GetSpec(),
                                                tmp.data());
    batch_fuse_ctx->UpdateTensorMap(const_in, input);
    batch_fuse_ctx->UpdateTensorBatchFuseMap(input, const_in);
    batch_fuse_ctx->UpdatePadInferShape(const_in, const_in->GetShape());
    batch_fuse_ctx->UpdateForwardPad(const_in, {0, 0, 0, 0});
    batch_fuse_ctx->UpdateForwardGap(const_in, {0, 0});
    tensor_queue.push_back(const_in);
    tensor_queue_pad_forward.push_back(const_in);
    tensor_clone_queue.push_back(const_in);
  }

  while (!tensor_queue_pad_forward.empty()) {
    //bfs
    auto tensor = tensor_queue_pad_forward.front();
    tensor_queue_pad_forward.pop_front();

    //dfs
    //const auto& tensor = tensor_queue_pad_forward.back();
    //tensor_queue_pad_forward.pop_back();q

    const auto& consumers = clone_batch_graph->GetConsumersOp(tensor);
    for (const auto& op : consumers) {
      if (op->impl()->kind_ != -1 && batch_fuse_ctx->IsReadyForPadInfer(op)) {
        auto tensors_pad_forward_infer = batch_fuse_impl::HandlePadInfernce(
            batch_fuse_ctx, op, false, true, false);  //forward pad infer
        if (tensors_pad_forward_infer.second) {
          //need_backward is true
          tensor_queue_pad_backward.push_back(tensor);
          while (!tensor_queue_pad_backward.empty()) {
            const auto& pad_tensor = tensor_queue_pad_backward.back();
            tensor_queue_pad_backward.pop_back();
            if (pad_tensor->GetSpec().attr_ == vx::TensorAttribute::INPUT) {
              for (const auto& t : tensors_pad_forward_infer.first) {
                tensor_queue_pad_forward.push_back(t);  //next_tensors
              }
            } else {
              const auto& producer =
                  clone_batch_graph->GetProducerOp(pad_tensor);
              auto tensors_pad_backward_infer =
                  batch_fuse_impl::HandlePadInfernce(batch_fuse_ctx, producer,
                                                     true, false, false);

              for (const auto& t : tensors_pad_backward_infer.first) {
                tensor_queue_pad_backward.push_back(t);  //former_tesnors
              }
              if (tensors_pad_backward_infer.first.empty()) {
                for (const auto& t : tensors_pad_forward_infer.first) {
                  tensor_queue_pad_forward.push_back(t);  //next_tensors
                }
              }
            }
          }
        } else {
          for (const auto& t : tensors_pad_forward_infer.first) {
            tensor_queue_pad_forward.push_back(t);  //next_tensors
          }
        }
      }
    }
  }

  while (!tensor_queue.empty()) {
    auto tensor = tensor_queue.front();
    tensor_queue.pop_front();
    const auto& consumers = clone_batch_graph->GetConsumersOp(tensor);
    for (const auto& op : consumers) {
      if (!batch_fuse_ctx->IsVisited(op) && op->impl()->kind_ != -1 &&
          batch_fuse_ctx->IsReadyForBatchFuse(op)) {
        auto next_tensors = batch_fuse_impl::HandleBatchFuse(
            batch_fuse_ctx, op, false, false, false);
        for (const auto& t : next_tensors) {
          tensor_queue.push_back(t);
        }
      }
    }
  }
  for (const auto& graph_input : batch_fuse_ctx->GetGraphInputMap()) {
    graph_io_map[graph_input.first] =
        batch_fuse_ctx->GetGraphInputMap()[graph_input.second];
  }
  for (const auto& graph_output : batch_fuse_ctx->GetGraphOutputMap()) {
    graph_io_map[graph_output.first] =
        batch_fuse_ctx->GetGraphOutputMap()[graph_output.second];
    auto out_shape = graph_output.second->GetShape();
    auto out_spec = graph_output.second->GetSpec();
  }
  return std::make_pair(batch_fuse_graph, graph_io_map);
}
}  // namespace fuse
}  // namespace tim
