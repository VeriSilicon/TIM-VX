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
#ifndef TIM_VX_OPS_CONV1D_H_
#define TIM_VX_OPS_CONV1D_H_

#include <array>

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

class Conv1d : public Operation {
 public:
  Conv1d(Graph* graph, PadType padding, uint32_t stride,
         uint32_t dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);
  Conv1d(Graph* graph, const std::array<uint32_t, 2>& pad,
         uint32_t stride, uint32_t dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);
  Conv1d(Graph* graph, int32_t weights, PadType padding,
         uint32_t ksize, uint32_t stride,
         uint32_t dilation, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);
  Conv1d(Graph* graph, int32_t weights, PadType padding,
         uint32_t ksize, uint32_t stride, uint32_t dilation,
         const std::array<uint32_t, 2>& pad, int32_t multiplier = 0,
         DataLayout input_layout = DataLayout::WHCN,
         DataLayout kernel_layout = DataLayout::WHIcOc);

  DataLayout KernelDataLayout() { return kernel_layout_; }

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const uint32_t weights_;
  const PadType padding_;
  const uint32_t ksize_;
  const uint32_t stride_;
  const uint32_t dilation_;
  const std::array<uint32_t, 2> pad_;
  const int32_t multiplier_;
  const DataLayout kernel_layout_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_CONV2D_H_ */