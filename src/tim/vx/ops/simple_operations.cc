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
#include "tim/vx/ops/simple_operations.h"

#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_SIMPLE_OP(NAME, VSI_OP_CODE)                             \
  NAME::NAME(Graph* graph) : Operation(graph, VSI_OP_CODE) {}           \
  std::shared_ptr<Operation> NAME::Clone(std::shared_ptr<Graph>& graph) \
      const {                                                           \
    return graph->CreateOperation<NAME>();                              \
  }

DEFINE_SIMPLE_OP(DataConvert, VSI_NN_OP_DATACONVERT)
DEFINE_SIMPLE_OP(Neg, VSI_NN_OP_NEG)
DEFINE_SIMPLE_OP(Abs, VSI_NN_OP_ABS)
DEFINE_SIMPLE_OP(Sin, VSI_NN_OP_SIN)
// TODO(jiangbo): enable it when ovxlib supports `Cos`
//DEFINE_SIMPLE_OP(Cos, VSI_NN_OP_COS)
DEFINE_SIMPLE_OP(Exp, VSI_NN_OP_EXP)
DEFINE_SIMPLE_OP(Log, VSI_NN_OP_LOG)
DEFINE_SIMPLE_OP(Sqrt, VSI_NN_OP_SQRT)
DEFINE_SIMPLE_OP(Rsqrt, VSI_NN_OP_RSQRT)
DEFINE_SIMPLE_OP(Square, VSI_NN_OP_SQUARE)
DEFINE_SIMPLE_OP(LogicalNot, VSI_NN_OP_LOGICAL_NOT)
DEFINE_SIMPLE_OP(Floor, VSI_NN_OP_FLOOR)
DEFINE_SIMPLE_OP(Cast, VSI_NN_OP_CAST)

#undef DEFINE_SIMPLE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim
