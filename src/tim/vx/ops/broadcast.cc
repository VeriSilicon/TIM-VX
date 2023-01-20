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
#include "tim/vx/ops/broadcast.h"

#include <cassert>
#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
Broadcast::Broadcast(Graph* graph, const std::vector<int32_t>& shape,
                     const std::vector<int32_t>& dimensions)
    : BuiltinOp(graph, VSI_NN_OP_EXPAND_BROADCAST),
      shape_(shape),
      dimensions_(dimensions) {
  this->impl()->node()->nn_param.expand_broadcast.dim_num = shape_.size();
  this->impl()->node()->nn_param.expand_broadcast.shape = (uint32_t*)shape_.data();
#ifdef VSI_EXPAND_BROADCAST_ENABLE_DIMENSIONS
  this->impl()->node()->nn_param.expand_broadcast.dimensions_num = dimensions_.size();
  if (dimensions.size() > 0)
  {
    int dim_num = shape.size();
    for (uint32_t i = 0; i < dimensions.size(); ++i) {
      dimensions_[i] += (dimensions[i] < 0 ? dim_num : 0U);
    }
    this->impl()->node()->nn_param.expand_broadcast.dimensions = (uint32_t*)dimensions_.data();
  } else {
    this->impl()->node()->nn_param.expand_broadcast.dimensions = nullptr;
  }
#endif
}

std::shared_ptr<Operation> Broadcast::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Broadcast>(this->shape_, this->dimensions_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim