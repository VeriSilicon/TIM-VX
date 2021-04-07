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
#ifndef TIM_VX_OPS_REDUCE_H_
#define TIM_VX_OPS_REDUCE_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

#define DECLARE_REDUCE_OP(NAME)                                  \
  class Reduce##NAME : public Operation {                        \
   public:                                                       \
    Reduce##NAME(Graph* graph, const std::vector<int32_t>& axis, \
                 bool keep_dims);                                \
                                                                 \
   protected:                                                    \
    std::vector<int32_t> axis_;                                  \
    bool keep_dims_;                                             \
  };

DECLARE_REDUCE_OP(Min);
DECLARE_REDUCE_OP(Max);
DECLARE_REDUCE_OP(Any);
DECLARE_REDUCE_OP(Prod);
DECLARE_REDUCE_OP(Mean);

#undef DECLARE_REDUCE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ACTIVATIONS_H_ */
