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
#ifndef TIM_VX_OPS_MAXUNPOOL2D_H_
#define TIM_VX_OPS_MAXUNPOOL2D_H_

#include <array>

#include "tim/vx/operation.h"
#include "tim/vx/types.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## MaxUnpool2d
 *
 * Performs an 2-D Max pooling operation upsample 
 *
 * - stride : stride along each spatial axis.
 * - ksize : filter size.
 */

class MaxUnpool2d : public Operation {
 public:
  MaxUnpool2d(Graph* graph, const std::array<uint32_t, 2>& ksize,
         const std::array<uint32_t, 2>& stride, DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const std::array<uint32_t, 2> ksize_;
  const std::array<uint32_t, 2> stride_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_MAXUNPOOL2D_H_ */
