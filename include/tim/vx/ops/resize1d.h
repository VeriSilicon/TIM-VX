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
#ifndef TIM_VX_OPS_RESIZE1D_H_
#define TIM_VX_OPS_RESIZE1D_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Resize1d
 *
 * Resize1ds 1D tensors to given size.
 *
 * - type : NEAREST_NEIGHBOR, BILINEAR or AREA.
 * - factor : scale the input size. DO NOT use it with target_height / target_width together.
 * - align_corners : If True, the centers of the 4 corner pixels of the input and output
 * tensors are aligned, preserving the values at the corner pixels.
 * - half_pixel_centers : If True, the pixel centers are assumed to be at (0.5, 0.5).
 * This is the default behavior of image.resize in TF 2.0. If this parameter is True,
 * then align_corners parameter must be False.
 * - target_height / target_width : output height / width. DO NOT use it with factor together.
 */

class Resize1d : public Operation {
 public:
  Resize1d(Graph* graph, ResizeType type, float factor, bool align_corners,
         bool half_pixel_centers, int target_size,
         DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const ResizeType type_;
  const float factor_;
  const bool align_corners_;
  const bool half_pixel_centers_;
  const int target_size_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_RESIZE1D_H_ */
