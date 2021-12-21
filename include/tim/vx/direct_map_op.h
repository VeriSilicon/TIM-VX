#ifndef TIM_VX_DIRECTMAPOP_H
#define TIM_VX_DIRECTMAPOP_H

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
// interface
class DirectMapOp : public Operation {
  public:
  DirectMapOp(Graph* graph, uint32_t kind, int in_cnt = 0, int out_cnt = 0,
              DataLayout layout = DataLayout::ANY);
};

}  // namespace vx

}  // namespace tim

#endif
