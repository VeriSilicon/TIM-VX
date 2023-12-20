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
#ifndef TIM_VX_OPS_CONV3D_H_
#define TIM_VX_OPS_CONV3D_H_

#include <array>
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Conv3d
 *
 * Performs a 3-D convolution operation
 * 
 * Input:
 * - input [WHDCN].
 * - kernel [ WHDIcOc ] (Ic: Input Channels. Oc: Output Channels).
 * - bias [ O ]. Optional.
 *
 * Attribute:
 * - weights : the output channel number for weight tensor.
 * - ksize : the height and width for weight tensor.
 * - padding : AUTO, VALID or SAME.
 * - pad : pad value for each spatial axis. (left, right, top, bottom, front, rear).
 * - stride : stride along each spatial axis.
 * - dilation : dilation value along each spatial axis of the filter.
 * - multiplier: function similar to group attribute on other framework,
 * but the value is different. multiplier = weights / group.
 * - input_layout : WHDCN or WHCDN.
 * - kernel_layout : WHDIcOc
 */

class Conv3d : public BuiltinOp {
 public:
  Conv3d(Graph* graph, PadType padding,
         const std::array<int32_t, 3>& stride,
         const std::array<int32_t, 3>& dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHDCN,
         DataLayout kernel_layout = DataLayout::WHDIcOc);
  Conv3d(Graph* graph, const std::array<int32_t, 6> pad,
         const std::array<int32_t, 3>& stride,
         const std::array<int32_t, 3>& dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHDCN,
         DataLayout kernel_layout = DataLayout::WHDIcOc);
  Conv3d(Graph* graph, int32_t weights, PadType padding,
         const std::array<int32_t, 3>& ksize,
         const std::array<int32_t, 3>& stride,
         const std::array<int32_t, 3>& dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHDCN,
         DataLayout kernel_layout = DataLayout::WHDIcOc);
  Conv3d(Graph* graph, int32_t weights, PadType padding,
         const std::array<int32_t, 3>& ksize,
         const std::array<int32_t, 3>& stride,
         const std::array<int32_t, 3>& dilation,
         const std::array<int32_t, 6>& pad, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHDCN,
         DataLayout kernel_layout = DataLayout::WHDIcOc);

  DataLayout KernelDataLayout() { return kernel_layout_; }

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const int32_t weights_;
  const PadType padding_;
  const std::array<int32_t, 3> ksize_;
  const std::array<int32_t, 3> stride_;
  const std::array<int32_t, 3> dilation_;
  const std::array<int32_t, 6> pad_;
  const int32_t multiplier_;
  const DataLayout kernel_layout_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_CONV3D_H_ */