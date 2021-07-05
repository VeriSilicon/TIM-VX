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
#include "tim/vx/ops/slice.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Slice::Slice(Graph* graph, uint32_t dims, const std::vector<int32_t>& start,
             const std::vector<int32_t>& length)
    : Operation(graph, VSI_NN_OP_SLICE),
      dims_(dims),
      start_(start),
      length_(length) {
  this->impl()->node()->nn_param.slice.dims = dims_;
  this->impl()->node()->nn_param.slice.start =
      reinterpret_cast<const uint32_t*>(start_.data());
  this->impl()->node()->nn_param.slice.length =
      reinterpret_cast<const uint32_t*>(length_.data());
}

std::shared_ptr<Operation> Slice::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Slice>(this->dims_, this->start_, this->length_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim