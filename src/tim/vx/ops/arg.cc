/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/arg.h"

#include "vsi_nn_pub.h"

#include "operation_private.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_ARG_OP(NAME, VSI_OP_TYPE, OP_PARAM)                           \
  Arg##NAME::Arg##NAME(Graph* graph, int32_t axis)                           \
      : Operation(graph, VSI_NN_OP_ARG##VSI_OP_TYPE), axis_(axis) {          \
    this->impl()->node()->nn_param.arg##OP_PARAM.axis = axis_;               \
  }                                                                          \
  std::shared_ptr<Operation> Arg##NAME::Clone(std::shared_ptr<Graph>& graph) \
      const {                                                                \
    return graph->CreateOperation<Arg##NAME>(this->axis_);                   \
  }

DEFINE_ARG_OP(Max, MAX, max);
DEFINE_ARG_OP(Min, MIN, min);
#undef DEFINE_ARG_OP
}  // namespace ops
}  // namespace vx
}  // namespace tim
