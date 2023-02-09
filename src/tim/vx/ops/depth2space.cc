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
#include "tim/vx/ops/depth2space.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

DepthToSpace::DepthToSpace(Graph* Graph, int block_size, DataLayout layout)
    : DepthToSpace(Graph, block_size, DCR_mode, layout) {}

DepthToSpace::DepthToSpace(Graph* Graph, int block_size, depth2space_mode mode,
                           DataLayout layout)
    : BuiltinOp(Graph, VSI_NN_OP_DEPTH2SPACE, 0, 0, layout),
      block_size_(block_size),
      mode_(mode) {
  this->impl()->node()->nn_param.depth2space.block_size = block_size_;
  this->impl()->node()->nn_param.depth2space.mode =
      (vsi_nn_depth2space_mode_e)mode_;
}

std::shared_ptr<Operation> DepthToSpace::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<DepthToSpace>(this->block_size_, this->mode_,
                                              this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim