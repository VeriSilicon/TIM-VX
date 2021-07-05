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
#include "tim/vx/ops/pad.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
Pad::Pad(Graph* graph, const std::vector<uint32_t>& front_size,
         const std::vector<uint32_t>& back_size, int32_t const_val)
    : Operation(graph, VSI_NN_OP_PAD),
      front_size_(front_size),
      back_size_(back_size),
      const_val_(const_val) {
  this->impl()->node()->nn_param.pad.front_size = front_size_.data();
  this->impl()->node()->nn_param.pad.back_size = back_size_.data();
  this->impl()->node()->nn_param.pad.dim_num = front_size_.size();
  this->impl()->node()->nn_param.pad.const_val = const_val_;
  this->impl()->node()->nn_param.pad.mode = VSI_NN_PAD_MODE_CONSTANT;
}

std::shared_ptr<Operation> Pad::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Pad>(this->front_size_, this->back_size_, this->const_val_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim