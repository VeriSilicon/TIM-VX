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
#ifndef TIM_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFERENCE_H_

#include <map>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/operation.h"
#include "src/tim/layout_infer/permute_vector.h"

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

std::vector<std::shared_ptr<vx::Tensor>> HandleLayoutInfer(
    std::shared_ptr<layout_inference_impl::LayoutInferContext>& ctx,
    const std::shared_ptr<vx::Operation>& op);
}  // namespace layout_inference_impl

std::pair<std::shared_ptr<vx::Graph>, /* infer graph */
          std::map<std::shared_ptr<vx::Tensor>,
                   std::shared_ptr<vx::Tensor>> /* graph io tensor map */>
LayoutInference(const std::shared_ptr<vx::Graph>& src_graph,
                std::shared_ptr<vx::Context>& ctx);

}  // namespace transform
}  // namespace tim

#endif