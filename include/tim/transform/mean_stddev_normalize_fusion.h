#ifndef TIM_MEAN_STD_DEV_NORMALIZE_FUSION_H
#define TIM_MEAN_STD_DEV_NORMALIZE_FUSION_H

#include <map>
#include <vector>
#include <memory>

namespace tim {
namespace vx {
class Context;
class Graph;
class Tensor;
class Operation;
}  // namespace vx

namespace transform {
void MeanStdDevNormalization(std::shared_ptr<vx::Graph>& src_graph);

}  // namespace transform
}  // namespace tim

#endif