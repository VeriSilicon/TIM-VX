#ifndef TIM_VX_BATCH_FUSE_CONTEXT_H_
#define TIM_VX_BATCH_FUSE_CONTEXT_H_
#include "tim/fuse/batch_fuse.h"

namespace tim {
namespace fuse {
using ShapeType = std::vector<uint32_t>;
namespace batch_fuse_impl {
class BatchFuseContext {
 public:
  BatchFuseContext(const std::shared_ptr<vx::Graph>& src_graph,
                   std::shared_ptr<vx::Graph>& batch_fuse_graph,
                   std::shared_ptr<vx::Graph>& clone_fuse_graph)
      : src_graph_(src_graph),
        batch_fuse_graph_(batch_fuse_graph),
        clone_batch_graph_(clone_fuse_graph) {}

  void MarkVisited(const std::shared_ptr<vx::Operation>& op);

  bool IsVisited(const std::shared_ptr<vx::Operation>& op) const;

  bool IsReadyForBatchFuse(const std::shared_ptr<vx::Operation>& op) const;

  bool IsReadyForGapInfer(const std::shared_ptr<vx::Operation>& op) const;

  bool IsReadyForCloneGraph(const std::shared_ptr<vx::Operation>& op) const;

  void UpdateTensorMap(const std::shared_ptr<vx::Tensor>& t_src,
                       const std::shared_ptr<vx::Tensor>& t_batch_fuse);

  void UpdateCloneTensorMap(const std::shared_ptr<vx::Tensor>& t_src,
                            const std::shared_ptr<vx::Tensor>& t_clone_graph);

  void UpdateForwardGap(const std::shared_ptr<vx::Tensor>& t_src,
                        const std::array<uint32_t, 2>& pad);

  void UpdateGapInferShape(const std::shared_ptr<vx::Tensor>& t_src,
                           const ShapeType& shape);

  void UpdateProportion(const std::shared_ptr<vx::Tensor>& t_src,
                        const float proportion);

  void UpdatePermAxisMap(const std::shared_ptr<vx::Tensor>& t_src,
                         const std::vector<uint32_t>& p_a_m);

  std::shared_ptr<vx::Tensor> GetMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::shared_ptr<vx::Tensor> GetCloneMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::array<uint32_t, 2> GetForwardGap(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  ShapeType GetGapInferShape(const std::shared_ptr<vx::Tensor>& t_src) const;

  float GetProportion(const std::shared_ptr<vx::Tensor>& t_src) const;

  std::vector<uint32_t> GetPermAxisMap(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  uint32_t GetFakeBatch() { return fake_batch_; }

  uint32_t GetBatchAxis() { return batch_axis_; }

  std::vector<uint32_t> GetFuseAxes() { return fuse_axes_; }

  uint32_t GetPermBatchAxis(const std::shared_ptr<vx::Tensor>& t_src) {
    // The index of batch axis
    auto perm_axis = perm_axis_map_[t_src];
    for (uint32_t i(0); i < perm_axis.size(); i++) {
      if (perm_axis[i] == batch_axis_) {
        return i;
      }
    }
    return 0;
  }

  uint32_t GetPermChannelAxis(const std::shared_ptr<vx::Tensor>& t_src) {
    // The index of channel axis
    auto perm_axis = perm_axis_map_[t_src];
    for (uint32_t i(0); i < perm_axis.size(); i++) {
      if (perm_axis[i] == channel_axis_) {
        return i;
      }
    }
    return 0;
  }

  std::vector<uint32_t> GetPermFuseAxes(
      const std::shared_ptr<vx::Tensor>& t_src) {
    // The index of fuse axes
    auto perm_axis = perm_axis_map_[t_src];
    std::vector<uint32_t> perm_fuse_axes;
    for (uint32_t i(0); i < perm_axis.size(); i++) {
      for (uint32_t j(0); j < fuse_axes_.size(); j++) {
        if (perm_axis[i] == fuse_axes_[j]) {
          perm_fuse_axes.push_back(i);
        }
      }
    }
    return perm_fuse_axes;
  }

  void SetFakeBatch(uint32_t batch) { fake_batch_ = batch; }

  void SetBatchAxis(uint32_t batch) { batch_axis_ = batch; }

  void SetChannelAxis(uint32_t channel) { channel_axis_ = channel; }

  void SetFuseAxes(const std::vector<uint32_t>& fuse_axes) {
    fuse_axes_ = fuse_axes;
  }

  const std::shared_ptr<vx::Graph>& GetSrcGraph() { return src_graph_; }
  std::shared_ptr<vx::Graph>& GetBatchFuseGraph() { return batch_fuse_graph_; }
  std::shared_ptr<vx::Graph>& GetCloneBatchGraph() { return clone_batch_graph_; }

 private:
  // After graph layout infered, we get a layout infered graph
  // Then we clone a new graph from layout infered graph, when we clone,
  // we just clone ops and set new shape for tensors in cloned graph, here we just multiply fake_batch_ times of batch
  // Lastly, we batch fuse the cloned graph to get a batch fused graph

  const std::shared_ptr<vx::Graph>& src_graph_;   //layout infered graph
  std::shared_ptr<vx::Graph>& batch_fuse_graph_;  //batch fused gaph
  std::shared_ptr<vx::Graph>&
      clone_batch_graph_;  //cloned from src_graph which can set fake batch

  std::vector<std::shared_ptr<vx::Operation>> visited_op_;

  // src graph is relatively, layout infered graph is the source graph of clone graph
  // clone graph is the source graph of batch fuse graph

  //tensor_in_src -> tensor_in_clone
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      clone_tensor_map_;
  // tensor_in_clone -> tensor_in_batch_fuse
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      tensor_map_;
  // tensor_in_src -> gap inside
  std::map<std::shared_ptr<vx::Tensor>, std::array<uint32_t, 2>>
      forward_gap_map_;
  // tensor_in_src -> fused_tensor for pad inference
  std::map<std::shared_ptr<vx::Tensor>, ShapeType> gap_infer_shape_map_;

  // tensor_in_src -> permute_axis_0, permute_axis_1
  std::map<std::shared_ptr<vx::Tensor>, std::vector<uint32_t>> perm_axis_map_;

  // Proportion of useful or valid data
  std::map<std::shared_ptr<vx::Tensor>, float> proportion_map_;

  uint32_t fake_batch_;
  uint32_t batch_axis_;
  uint32_t channel_axis_;
  std::vector<uint32_t> fuse_axes_;
};

}  // namespace batch_fuse_impl
}  // namespace fuse
}  // namespace tim

#endif