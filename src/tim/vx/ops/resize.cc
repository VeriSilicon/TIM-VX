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
#include "tim/vx/ops/resize.h"

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Resize::Resize(Graph* graph, ResizeType type, float factor, bool align_corners,
               bool half_pixel_centers, int target_height, int target_width,
               DataLayout layout)
    : Operation(graph, VSI_NN_OP_RESIZE, 0, 0, layout),
      type_(type),
      factor_(factor),
      align_corners_(align_corners),
      half_pixel_centers_(half_pixel_centers),
      target_height_(target_height),
      target_width_(target_width) {
  impl()->node()->nn_param.resize.type = TranslateResizeType(type);
  impl()->node()->nn_param.resize.factor = factor;
  impl()->node()->nn_param.resize.align_corners = ToVxBool(align_corners);
  impl()->node()->nn_param.resize.half_pixel_centers =
      ToVxBool(half_pixel_centers);
  impl()->node()->nn_param.resize.size[0] = target_width;
  impl()->node()->nn_param.resize.size[1] = target_height;
}

std::shared_ptr<Operation> Resize::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Resize>(
      this->type_, this->factor_, this->align_corners_,
      this->half_pixel_centers_, this->target_height_, this->target_width_,
      this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
