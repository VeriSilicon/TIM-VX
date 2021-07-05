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

/**
 * ## ReduceMin
 *
 * Reduces a tensor by computing the minimum of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceMax
 *
 * Reduces a tensor by computing the maximum of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceAny
 *
 * Reduces a tensor by computing the "logical or" of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceAll
 *
 * Reduces a tensor by computing the "logical and" of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceProd
 *
 * Reduces a tensor by computing the multiplying of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceMean
 *
 * Reduces a tensor by computing the mean of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 *
 * ## ReduceSum
 *
 * Reduces a tensor by computing the summing of elements along given dimensions.
 *
 * - axis : the dimensions to reduce.
 * - keep_dims : If keep_dims is true, the reduced dimensions are retained with
 * length 1. Otherwise, the rank of the tensor is reduced by 1 for each entry
 * in dimensions
 */

#define DECLARE_REDUCE_OP(NAME)                                  \
  class Reduce##NAME : public Operation {                        \
   public:                                                       \
    Reduce##NAME(Graph* graph, const std::vector<int32_t>& axis, \
                 bool keep_dims);                                \
  std::shared_ptr<Operation>                                     \
    Clone(std::shared_ptr<Graph>& graph) const override;         \
                                                                 \
   protected:                                                    \
    std::vector<int32_t> axis_;                                  \
    bool keep_dims_;                                             \
  };

DECLARE_REDUCE_OP(Min);
DECLARE_REDUCE_OP(Max);
DECLARE_REDUCE_OP(Any);
DECLARE_REDUCE_OP(All);
DECLARE_REDUCE_OP(Prod);
DECLARE_REDUCE_OP(Mean);
DECLARE_REDUCE_OP(Sum);

#undef DECLARE_REDUCE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ACTIVATIONS_H_ */
