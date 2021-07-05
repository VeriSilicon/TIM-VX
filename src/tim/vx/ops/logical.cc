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
#include "tim/vx/ops/logical.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_LOGICAL_OP(NAME, VSI_OP_CODE)           \
  Logical##NAME::Logical##NAME(Graph* graph)           \
      : Operation(graph, VSI_NN_OP_LOGICAL_OPS) {      \
    this->impl()->node()->nn_param.relational_ops.op = \
        VSI_NN_LOGICAL_##VSI_OP_CODE;                  \
  }                                                    \
  std::shared_ptr<Operation> Logical##NAME::Clone(     \
      std::shared_ptr<Graph>& graph) const {           \
    return graph->CreateOperation<Logical##NAME>();    \
  }

DEFINE_LOGICAL_OP(And, AND);
DEFINE_LOGICAL_OP(Or, OR);

#undef DEFINE_LOGICAL_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim