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
#ifndef TIM_VX_OPS_L2NOMALIZATION_H_
#define TIM_VX_OPS_L2NOMALIZATION_H_
#include "tim/vx/operation.h"

/**
 * ## L2Normalization
 *
 * Applies L2 normalization along the axis dimension:
 *
 * ```
 * output[batch, row, col, channel] =
 *  input[batch, row, col, channel] /
 *  sqrt(sum_{c} pow(input[batch, row, col, c], 2))
 * ```
 */

namespace tim {
namespace vx {
namespace ops {
class L2Normalization : public Operation {
 public:
  L2Normalization(Graph* graph, int32_t axis);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  int32_t axis_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif
