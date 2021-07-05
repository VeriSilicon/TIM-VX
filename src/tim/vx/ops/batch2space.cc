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
#include "tim/vx/ops/batch2space.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Batch2Space::Batch2Space(Graph* graph, const std::vector<int>& block_size,
                         const std::vector<int>& crop, DataLayout layout)
    : Operation(graph, VSI_NN_OP_BATCH2SPACE, 0, 0, layout),
      block_size_(block_size),
      crop_(crop) {
  this->impl()->node()->nn_param.batch2space.block_size = block_size_.data();
  this->impl()->node()->nn_param.batch2space.block_size_num = block_size.size();
  // the size of crop_ should be 4.
  for (size_t i = 0; i < crop_.size(); i++) {
    this->impl()->node()->nn_param.batch2space.crop[i] = crop_[i];
  }
}

std::shared_ptr<Operation> Batch2Space::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Batch2Space>(this->block_size_, this->crop_,
                                             this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
