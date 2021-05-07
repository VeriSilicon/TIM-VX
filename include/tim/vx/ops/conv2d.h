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
#ifndef TIM_VX_OPS_CONV2D_H_
#define TIM_VX_OPS_CONV2D_H_

#include <array>

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

class Conv2d : public Operation {
 public:
  Conv2d(Graph* graph, int32_t weights, PadType padding,
         const std::array<uint32_t, 2>& ksize,
         const std::array<uint32_t, 2>& stride,
         const std::array<uint32_t, 2>& dilation, int32_t multiplier = 0,
         DataLayout layout = DataLayout::WHCN);
  Conv2d(Graph* graph, int32_t weights, PadType padding,
         const std::array<uint32_t, 2>& ksize,
         const std::array<uint32_t, 2>& stride,
         const std::array<uint32_t, 2>& dilation,
         const std::array<uint32_t, 4>& pad, int32_t multiplier = 0,
         DataLayout layout = DataLayout::WHCN);

 protected:
  const uint32_t weights_;
  const PadType padding_;
  const std::array<uint32_t, 2> ksize_;
  const std::array<uint32_t, 2> stride_;
  const std::array<uint32_t, 2> dilation_;
  const std::array<uint32_t, 4> pad_;
  const int32_t multiplier_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_CONV2D_H_ */