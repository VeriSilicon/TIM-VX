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
#include "tim/vx/ops/clip.h"

#include "vsi_nn_pub.h"

#include "operation_private.h"

namespace tim {
namespace vx {
namespace ops {


Clip::Clip(Graph* graph, float min, float max)
  : Operation(graph, VSI_NN_OP_CLIP),
    min_(min),
    max_(max) {
  this->impl()->node()->nn_param.clip.min = min_;
  this->impl()->node()->nn_param.clip.max = max_;
}

std::shared_ptr<Operation> Clip::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Clip>(this->impl_->node()->nn_param.clip.min,
                                      this->impl_->node_->nn_param.clip.max);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
