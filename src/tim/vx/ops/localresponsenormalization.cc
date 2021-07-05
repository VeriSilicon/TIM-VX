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
#include "tim/vx/ops/localresponsenormalization.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
LocalResponseNormalization::LocalResponseNormalization(Graph* graph,
                                                       uint32_t size,
                                                       float alpha, float beta,
                                                       float bias, int32_t axis)
    : Operation(graph, VSI_NN_OP_LRN2),
      size_(size),
      alpha_(alpha),
      beta_(beta),
      bias_(bias),
      axis_(axis) {
  this->impl()->node()->nn_param.lrn.size = size_ * 2 + 1;
  this->impl()->node()->nn_param.lrn.alpha = alpha_;
  this->impl()->node()->nn_param.lrn.beta = beta_;
  this->impl()->node()->nn_param.lrn.bias = bias_;
  this->impl()->node()->nn_param.lrn.axis = axis_;
  this->impl()->node()->nn_param.lrn.type =
      VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS;
}

std::shared_ptr<Operation> LocalResponseNormalization::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<LocalResponseNormalization>(
      this->size_, this->alpha_, this->beta_, this->bias_, this->axis_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim