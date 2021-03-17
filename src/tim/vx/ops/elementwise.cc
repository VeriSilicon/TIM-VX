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
#include "tim/vx/ops/elementwise.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_ELEMENTWISE_UNARY_OP(NAME, VSI_OP_CODE) \
  NAME::NAME(Graph* graph) : Operation(graph, VSI_OP_CODE) {}

DEFINE_ELEMENTWISE_UNARY_OP(Abs, VSI_NN_OP_ABS);
DEFINE_ELEMENTWISE_UNARY_OP(Sin, VSI_NN_OP_SIN);
// TODO(jiangbo): enable it when ovxlib supports `Cos`
//DEFINE_ELEMENTWISE_UNARY_OP(Cos, VSI_NN_OP_COS);
DEFINE_ELEMENTWISE_UNARY_OP(Exp, VSI_NN_OP_EXP);
DEFINE_ELEMENTWISE_UNARY_OP(Log, VSI_NN_OP_LOG);
DEFINE_ELEMENTWISE_UNARY_OP(Sqrt, VSI_NN_OP_SQRT);
DEFINE_ELEMENTWISE_UNARY_OP(Rsqrt, VSI_NN_OP_RSQRT);
DEFINE_ELEMENTWISE_UNARY_OP(Square, VSI_NN_OP_SQUARE);
DEFINE_ELEMENTWISE_UNARY_OP(LogicalNot, VSI_NN_OP_LOGICAL_NOT);

#undef DEFINE_ELEMENTWISE_UNARY_OP

#define DEFINE_ELEMENTWISE_BINARY_OP(NAME, VSI_OP_CODE) \
  NAME::NAME(Graph* graph) : Operation(graph, VSI_OP_CODE, 2, 1) {}

DEFINE_ELEMENTWISE_BINARY_OP(Minimum, VSI_NN_OP_MINIMUM);
DEFINE_ELEMENTWISE_BINARY_OP(Maximum, VSI_NN_OP_MAXIMUM);
DEFINE_ELEMENTWISE_BINARY_OP(Add, VSI_NN_OP_ADD);
DEFINE_ELEMENTWISE_BINARY_OP(Sub, VSI_NN_OP_SUBTRACT);
DEFINE_ELEMENTWISE_BINARY_OP(Div, VSI_NN_OP_DIVIDE);
DEFINE_ELEMENTWISE_BINARY_OP(Pow, VSI_NN_OP_POW);

#undef DEFINE_ELEMENTWISE_BINARY_OP

Multiply::Multiply(Graph* graph) : Operation(graph, VSI_NN_OP_MULTIPLY, 2, 1) {
    this->impl()->node()->nn_param.multiply.scale = 1.0f;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim
