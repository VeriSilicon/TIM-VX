#ifndef TIM_VX_BATCH_FUSE_CONTEXT_H_
#define TIM_VX_BATCH_FUSE_CONTEXT_H_
// #include "permute_vector.h"
#include "tim/fuse/batch_fuse.h"

namespace tim {
namespace fuse {
using ShapeType = std::vector<uint32_t>;
namespace batch_fuse_impl {
class BatchFuseContext {
 public:
  BatchFuseContext(const std::shared_ptr<vx::Graph>& src_graph,
                   std::shared_ptr<vx::Graph>& batch_fuse_graph, std::shared_ptr<vx::Graph>& clone_fuse_graph)
      : src_graph_(src_graph), batch_fuse_graph_(batch_fuse_graph), clone_batch_graph_(clone_fuse_graph) {}
  //   void SetPermuteVector(std::shared_ptr<vx::Tensor> tensor,
  //                         std::shared_ptr<IPermuteVector> pv);
  //   const std::shared_ptr<IPermuteVector> GetPermuteVector(
  //       const std::shared_ptr<vx::Tensor>& tensor) const;
  void MarkVisited(const std::shared_ptr<vx::Operation>& op);
  bool IsVisited(const std::shared_ptr<vx::Operation>& op) const;
  bool IsReadyForBatchFuse(const std::shared_ptr<vx::Operation>& op) const;
  bool IsReadyForPadInfer(const std::shared_ptr<vx::Operation>& op) const;
  bool IsReadyForCloneGraph(const std::shared_ptr<vx::Operation>& op) const;
  void UpdateTensorMap(const std::shared_ptr<vx::Tensor>& t_src,
                       const std::shared_ptr<vx::Tensor>& t_batch_fuse);
  void UpdateCloneTensorMap(const std::shared_ptr<vx::Tensor>& t_src,
                       const std::shared_ptr<vx::Tensor>& t_clone_graph);
  void UpdateTensorBatchFuseMap(const std::shared_ptr<vx::Tensor>& t_batch_fuse,
                                const std::shared_ptr<vx::Tensor>& t_src);
  void UpdateInitPad(const std::shared_ptr<vx::Tensor>& t_src,
                     const std::array<uint32_t, 4>& pad);
  void UpdateBackwardPad(const std::shared_ptr<vx::Tensor>& t_src,
                         const std::array<uint32_t, 4>& pad);
  void UpdateForwardPad(const std::shared_ptr<vx::Tensor>& t_src,
                        const std::array<uint32_t, 4>& pad);

  void UpdatePadInferShape(const std::shared_ptr<vx::Tensor>& t_src,
                        const ShapeType& shape);

  void UpdateProportion(const std::shared_ptr<vx::Tensor>& t_src,
                        const int32_t proportion);

  std::shared_ptr<vx::Tensor> GetMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::shared_ptr<vx::Tensor> GetCloneMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::shared_ptr<vx::Tensor> GetBatchFuseMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_batch_fuse) const;

  std::array<uint32_t, 4> GetInitPad(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::array<uint32_t, 4> GetBackwardPad(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  std::array<uint32_t, 4> GetForwardPad(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  ShapeType GetPadInferShape(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  int32_t GetProportion(const std::shared_ptr<vx::Tensor>& t_src) const;

  void UpdateGraphInputMap(const std::shared_ptr<vx::Tensor>& i_src,
                           const std::shared_ptr<vx::Tensor>& i_batch_fuse);

  void UpdateGraphOutputMap(const std::shared_ptr<vx::Tensor>& o_src,
                            const std::shared_ptr<vx::Tensor>& o_batch_fuse);

  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
  GetGraphInputMap() const {
    return graph_input_map_;
  }

  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
  GetGraphOutputMap() const {
    return graph_output_map_;
  }

  uint32_t GetFakeBatch(){
    return fake_batch_;
  }

  void SetFakeBatch(uint32_t batch){
    fake_batch_ = batch;
  }

  const std::shared_ptr<vx::Graph>& src_graph_;
  std::shared_ptr<vx::Graph>& batch_fuse_graph_;
  std::shared_ptr<vx::Graph>& clone_batch_graph_;

 private:
  //   std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<IPermuteVector>>
  //       tensor_pv_;
  std::vector<std::shared_ptr<vx::Operation>> visited_op_;
  // tensor_in_src -> tensor_in_layout
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      tensor_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      clone_tensor_map_;
  // tensor_bacth_fused -> tensor_in_src
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      tensor_batch_fuse_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_input_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_output_map_;

  //pad inference
  //init pad size for op need
  std::map<std::shared_ptr<vx::Tensor>, std::array<uint32_t, 4>> init_pad_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::array<uint32_t, 4>>
      backward_pad_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::array<uint32_t, 4>>
      forward_pad_map_;

  //t_src -> fused_tensor for pad inference
  std::map<std::shared_ptr<vx::Tensor>, ShapeType>
      pad_infer_shape_map_;

  //Proportion of useful data
  std::map<std::shared_ptr<vx::Tensor>, int32_t> proportion_map_;

  uint32_t fake_batch_;
};

}  // namespace batch_fuse_impl
}  // namespace fuse
}  // namespace tim

#endif