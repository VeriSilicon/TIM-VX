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
#include "tim/vx/ops/resize1d.h"

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Resize1d::Resize1d(Graph* graph, ResizeType type, float factor, bool align_corners,
               bool half_pixel_centers, int target_size, DataLayout layout)
    : Operation(graph, VSI_NN_OP_RESIZE_1D, 0, 0, layout),
      type_(type),
      factor_(factor),
      align_corners_(align_corners),
      half_pixel_centers_(half_pixel_centers),
      target_size_(target_size) {
  impl()->node()->nn_param.resize_1d.type = TranslateResizeType(type);
  impl()->node()->nn_param.resize_1d.factor = factor;
  impl()->node()->nn_param.resize_1d.align_corners = ToVxBool(align_corners);
  impl()->node()->nn_param.resize_1d.half_pixel_centers =
      ToVxBool(half_pixel_centers);
  impl()->node()->nn_param.resize_1d.size[0] = target_size;
}

std::shared_ptr<Operation> Resize1d::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Resize1d>(
      this->type_, this->factor_, this->align_corners_,
      this->half_pixel_centers_, this->target_size_, this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
