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
#ifndef TIM_VX_OPS_MAX_POOL3D_H_
#define TIM_VX_OPS_MAX_POOL3D_H_

#include "tim/vx/builtin_op.h"
#include "tim/vx/types.h"
#include <array>

#ifdef VSI_FEAT_OP_MAX_POOL3D

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Max_pool3d
 * 
 * Applies a 3D max pooling over an input Tensor which can be regarded as a composition of 3D planes.
 * 
 * Input:
 * - input [WHDCN]
 * - kernel [ WHD ] 
 * 
 * Attribute:
 * - round_type : CEILING or FLOOR
 * - ksize : the height and width for kernel tensor.
 * - stride : stride along each spatial axis.
 * - pad : pad value for each spatial axis. (left, right, top, bottom, front, rear).
 * - pad_type : AUTO, VALID or SAME.
 * 
 */

class MaxPool3d : public BuiltinOp {
 public:
  MaxPool3d(Graph* Graph, RoundType round_type,
            const std::array<uint32_t, 3>& ksize, 
            const std::array<uint32_t, 3>& stride,
            const std::array<uint32_t, 6>& pad,
            PadType pad_type,
            DataLayout layout = DataLayout::WHDCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;
  
 protected:
  const RoundType round_type_;
  const std::array<uint32_t, 3> ksize_;
  const std::array<uint32_t, 3> stride_;
  const std::array<uint32_t, 6> pad_;
  const PadType pad_type_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_MAX_POOL3D)
#endif /* TIM_VX_OPS_MAX_POOL3D_H_ */
