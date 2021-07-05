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

/**
 * ## Add
 *
 * Add(x, y) : x + y. This operation supports broadcasting.
 *
 * ## Sub
 *
 * Sub(x, y) : x - y. This operation supports broadcasting.
 *
 * ## Multiply
 *
 * Multiply(x, y) : Multiplies two tensors, element-wise, also known as Hadamard
 * product. This operation supports broadcasting.
 *
 * - scale: scaling the product.
 *
 * ## Div
 *
 * Div(x, y) : x / y. This operation supports broadcasting.
 *
 * ## Pow
 *
 * Pow(x, y) : x ^ y. This operation supports broadcasting.
 *
 * ## Minimum
 *
 * Minimum(x, y) : min(x, y). This operation supports broadcasting.
 *
 * ## Maximum
 *
 * Maximum(x, y) : max(x, y). This operation supports broadcasting.
 *
 * ## FloorDiv
 *
 * FloorDiv(x, y): floor( x / y ). This operation supports broadcasting.
 */

#define DECLARE_ELEMENTWISE_OP(NAME)                   \
  class NAME : public Operation {                      \
   public:                                             \
    NAME(Graph* graph);                                \
    std::shared_ptr<Operation> Clone(                  \
        std::shared_ptr<Graph>& graph) const override; \
  };

DECLARE_ELEMENTWISE_OP(Minimum)
DECLARE_ELEMENTWISE_OP(Maximum)
DECLARE_ELEMENTWISE_OP(Add)
DECLARE_ELEMENTWISE_OP(Sub)
DECLARE_ELEMENTWISE_OP(Div)
DECLARE_ELEMENTWISE_OP(Pow)
DECLARE_ELEMENTWISE_OP(FloorDiv)

class Multiply : public Operation {
 public:
  Multiply(Graph* graph, float scale = 1.0f);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;
};

#undef DECLARE_ELEMENTWISE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ELEMENTWISE_H_ */
