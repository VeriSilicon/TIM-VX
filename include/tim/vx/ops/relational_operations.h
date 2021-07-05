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
#ifndef TIM_VX_OPS_RELATIONAL_H_
#define TIM_VX_OPS_RELATIONAL_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Greater
 *
 * For input tensors x and y, computes x > y elementwise.
 *
 * ## GreaterOrEqual
 *
 * For input tensors x and y, computes x >= y elementwise.
 *
 * ## Less
 *
 * For input tensors x and y, computes x < y elementwise.
 *
 * ## LessOrEqual
 *
 * For input tensors x and y, computes x <= y elementwise.
 *
 * ## NotEqual
 *
 * For input tensors x and y, computes x != y elementwise.
 *
 * ## Equal
 *
 * For input tensors x and y, computes x == y elementwise.
 */

#define DECLARE_RELATIONAL_OP(NAME)                    \
  class NAME : public Operation {                      \
   public:                                             \
    NAME(Graph* graph);                                \
    std::shared_ptr<Operation> Clone(                  \
        std::shared_ptr<Graph>& graph) const override; \
  };

DECLARE_RELATIONAL_OP(Greater)
DECLARE_RELATIONAL_OP(GreaterOrEqual)
DECLARE_RELATIONAL_OP(Less)
DECLARE_RELATIONAL_OP(LessOrEqual)
DECLARE_RELATIONAL_OP(NotEqual)
DECLARE_RELATIONAL_OP(Equal)

#undef DECLARE_RELATIONAL_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_RELATIONAL_H_ */
