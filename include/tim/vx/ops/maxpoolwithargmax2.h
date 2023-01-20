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
#ifndef TIM_VX_OPS_MAXPOOLWITHARGMAX2_H_
#define TIM_VX_OPS_MAXPOOLWITHARGMAX2_H_

#include <array>

#include "tim/vx/builtin_op.h"
#include "tim/vx/types.h"

#ifdef VSI_FEAT_OP_MAXPOOLWITHARGMAX

namespace tim {
namespace vx {
namespace ops {

/**
 * ## MaxpoolWithArgmax2
 *
 * Performs an 2-D Max pooling operation and return indices(which start at the beginning of the input tensor).
 *
 * - padding : AUTO, VALID or SAME.
 * - ksize : filter size.
 * - stride : stride along each spatial axis.
 * - round_type : CEILING or FLOOR.
 */

class MaxpoolWithArgmax2 : public BuiltinOp {
 public:
  MaxpoolWithArgmax2(Graph* graph, PadType padding,
         const std::array<uint32_t, 2>& ksize,
         const std::array<uint32_t, 2>& stride,
         RoundType round_type = RoundType::FLOOR,
         DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const PadType padding_;
  const std::array<uint32_t, 2> ksize_;
  const std::array<uint32_t, 2> stride_;
  const RoundType round_type_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_MAXPOOLWITHARGMAX)
#endif /* TIM_VX_OPS_MAXPOOLWITHARGMAX2_H_ */
