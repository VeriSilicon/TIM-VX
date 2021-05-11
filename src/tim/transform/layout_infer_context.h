#ifndef TIM_VX_LAYOUT_INFER_CONTEXT_H_
#define TIM_VX_LAYOUT_INFER_CONTEXT_H_
#include "permute_vector.h"
#include "tim/transform/layout_inference.h"

namespace tim {
namespace transform {
namespace layout_inference_impl {
class LayoutInferContext {
 public:
  LayoutInferContext(const std::shared_ptr<vx::Graph>& src_graph,
                     std::shared_ptr<vx::Graph>& infer_graph)
      : src_graph_(src_graph), infer_graph_(infer_graph) {}
  void SetPermuteVector(std::shared_ptr<vx::Tensor> tensor,
                        std::shared_ptr<IPermuteVector> pv);
  const std::shared_ptr<IPermuteVector> GetPermuteVector(
      const std::shared_ptr<vx::Tensor>& tensor) const;
  void MarkVisited(const std::shared_ptr<vx::Operation>& op);
  bool IsVisited(const std::shared_ptr<vx::Operation>& op) const;
  bool IsReadyForInfer(const std::shared_ptr<vx::Operation>& op) const;
  void UpdateTensorMap(const std::shared_ptr<vx::Tensor>& t_src,
                       const std::shared_ptr<vx::Tensor>& t_layout);
  std::shared_ptr<vx::Tensor> GetMapedTensor(
      const std::shared_ptr<vx::Tensor>& t_src) const;

  void UpdateGraphInputMap(const std::shared_ptr<vx::Tensor>& i_src,
                           const std::shared_ptr<vx::Tensor>& i_layout);

  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
  GetGraphInputMap() const {
    return graph_input_map_;
  }

  const std::shared_ptr<vx::Graph>& src_graph_;
  std::shared_ptr<vx::Graph>& infer_graph_;

 private:
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<IPermuteVector>>
      tensor_pv_;
  std::vector<std::shared_ptr<vx::Operation>> visited_op_;
  // tensor_in_src -> tensor_in_layout
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      tensor_map_;
  std::map<std::shared_ptr<vx::Tensor>, std::shared_ptr<vx::Tensor>>
      graph_input_map_;
};

}  // namespace layout_inference_impl
}  // namespace transform
}  // namespace tim

#endif