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
#ifndef TIM_VX_OPS_GROUPEDCONV2D_H_
#define TIM_VX_OPS_GROUPEDCONV2D_H_

#include <array>

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## GroupedConv2d
 *
 * Performs a grouped 2-D convolution operation.
 * 
 * Input:
 * - input [WHCN or CWHN].
 * - kernel [ WHIcOc ] (Ic: Input Channels. Oc: Output Channels).
 * - bias [ O ]. Optional.
 *
 * Attribute:
 * - weights : the output channel number for weight tensor.
 * - ksize : the height and width for weight tensor.
 * - padding : AUTO, VALID or SAME.
 * - pad : pad value for each spatial axis.
 * - stride : stride along each spatial axis.
 * - dilation : dilation value along each spatial axis of the filter.
 * - group_number: Split conv to n group.
 * - layout : WHCN or CWHN.
 */

class GroupedConv2d : public Operation {
 public:
  GroupedConv2d(Graph* graph, PadType padding,
         const std::array<uint32_t, 2>& strides,
         const std::array<uint32_t, 2>& dilation,
         int32_t grouped_number,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);
  GroupedConv2d(Graph* graph,
         const std::array<uint32_t, 4>& pad,
         const std::array<uint32_t, 2>& strides,
         const std::array<uint32_t, 2>& dilation,
         int32_t group_number,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);

  DataLayout KernelDataLayout() { return kernel_layout_; }

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const PadType padding_;
  const std::array<uint32_t, 2> strides_;
  const std::array<uint32_t, 2> dilation_;
  const std::array<uint32_t, 4> pad_;
  const int32_t group_number_;
  const DataLayout kernel_layout_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_GROUPED_CONV2D_H_ */