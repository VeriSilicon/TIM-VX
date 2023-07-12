/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
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
#ifndef TIM_VX_OPS_TOPK_H_
#define TIM_VX_OPS_TOPK_H_

#include <array>

#include "tim/vx/builtin_op.h"
#include "tim/vx/types.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Topk
 *
 * Finds values and indices of the k largest entries for the last dimension.
 *
 * - k : Number of top elements to look for along the last dimension.
 * -axis : Dimension on which to do th sort. Default is 0.
 */

class Topk : public BuiltinOp {
 public:
  Topk(Graph* graph, uint32_t k, int32_t axis = 0);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_TOPK_H_ */