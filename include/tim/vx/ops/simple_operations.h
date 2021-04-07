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
#ifndef TIM_VX_OPS_SIMPLE_OPERATIONS_H_
#define TIM_VX_OPS_SIMPLE_OPERATIONS_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

#define DECLARE_SIMPLE_OP(NAME)   \
  class NAME : public Operation { \
   public:                        \
    NAME(Graph* graph);           \
  };

DECLARE_SIMPLE_OP(DataConvert)
DECLARE_SIMPLE_OP(Neg)
DECLARE_SIMPLE_OP(Abs)
DECLARE_SIMPLE_OP(Sin)
// TODO(jiangbo): enable it when internal ops supports `Cos`
//DECLARE_SIMPLE_OP(Cos)
DECLARE_SIMPLE_OP(Exp)
DECLARE_SIMPLE_OP(Log)
DECLARE_SIMPLE_OP(Sqrt)
DECLARE_SIMPLE_OP(Rsqrt)
DECLARE_SIMPLE_OP(Square)
DECLARE_SIMPLE_OP(LogicalNot)

#undef DECLARE_SIMPLE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_SIMPLE_OPERATIONS_H_ */
