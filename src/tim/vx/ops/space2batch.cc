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
#include "tim/vx/ops/space2batch.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Space2Batch::Space2Batch(Graph* graph, const std::vector<int>& block_size,
                         const std::vector<int>& pad, DataLayout layout)
    : Operation(graph, VSI_NN_OP_SPACE2BATCH, 0, 0, layout),
      block_size_(block_size),
      pad_(pad) {
  this->impl()->node()->nn_param.space2batch.block_size = block_size_.data();
  this->impl()->node()->nn_param.space2batch.block_size_num = block_size.size();
  // the size of pad_ should be 4.
  for (size_t i = 0; i < pad_.size(); i++) {
    this->impl()->node()->nn_param.space2batch.pad[i] = pad_[i];
  }
}

std::shared_ptr<Operation> Space2Batch::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Space2Batch>(this->block_size_, this->pad_,
                                             this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
