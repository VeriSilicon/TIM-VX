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
#include "tim/vx/ops/roi_align.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

RoiAlign::RoiAlign(Graph* graph, int32_t output_height, int32_t output_width,
          float height_ratio, float width_ratio, int32_t height_sample_num,
          int32_t width_sample_num, DataLayout input_layout)
    : BuiltinOp(graph, VSI_NN_OP_ROI_ALIGN, 0, 0, input_layout),
      output_height_(output_height),
      output_width_(output_width),
      height_ratio_(height_ratio),
      width_ratio_(width_ratio),
      height_sample_num_(height_sample_num),
      width_sample_num_(width_sample_num) {
  this->impl()->node()->nn_param.roi_align.output_height = output_height;
  this->impl()->node()->nn_param.roi_align.output_width = output_width;
  this->impl()->node()->nn_param.roi_align.height_ratio = height_ratio;
  this->impl()->node()->nn_param.roi_align.width_ratio = width_ratio;
  this->impl()->node()->nn_param.roi_align.height_sample_num =
      height_sample_num;
  this->impl()->node()->nn_param.roi_align.width_sample_num = width_sample_num;
}

std::shared_ptr<Operation> RoiAlign::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<RoiAlign>(
      this->output_height_, this->output_width_, this->height_ratio_,
      this->width_ratio_, this->height_sample_num_, this->width_sample_num_,
      this->impl_->layout_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim