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
#ifndef TIM_VX_OPS_ELEMENTWISE_H_
#define TIM_VX_OPS_ELEMENTWISE_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

#define DELCATE_ELEMENTWISE_OP(NAME) \
  class NAME : public Operation {    \
   public:                           \
    NAME(Graph* graph);              \
  };

DELCATE_ELEMENTWISE_OP(Abs)
DELCATE_ELEMENTWISE_OP(Sin)
// TODO(jiangbo): enable it when internal ops supports `Cos`
//DELCATE_ELEMENTWISE_OP(Cos)
DELCATE_ELEMENTWISE_OP(Exp)
DELCATE_ELEMENTWISE_OP(Log)
DELCATE_ELEMENTWISE_OP(Sqrt)
DELCATE_ELEMENTWISE_OP(Rsqrt)
DELCATE_ELEMENTWISE_OP(Square)
DELCATE_ELEMENTWISE_OP(LogicalNot)

DELCATE_ELEMENTWISE_OP(Minimum)
DELCATE_ELEMENTWISE_OP(Maximum)
DELCATE_ELEMENTWISE_OP(Add)
DELCATE_ELEMENTWISE_OP(Sub)
DELCATE_ELEMENTWISE_OP(Div)
DELCATE_ELEMENTWISE_OP(Pow)

class Multiply : public Operation {
  public:
    Multiply(Graph* graph, float scale = 1.0f);
};

#undef DELCATE_ELEMENTWISE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ELEMENTWISE_H_ */
