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
#ifndef TIM_VX_OPS_ROI_ALIGN_H_
#define TIM_VX_OPS_ROI_ALIGN_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## RoiAlign
 *
 * Select and scale the feature map of each region of interest to a unified output
 * size by average pooling sampling points from bilinear interpolation.
 *
 * - output_height : specifying the output height of the output tensor.
 * - output_width : specifying the output width of the output tensor.
 * - height_ratio : specifying the ratio from the height of original image to the
 *   height of feature map.
 * - width_ratio : specifying the ratio from the width of original image to the
 *   width of feature map.
 * - height_sample_num :  specifying the number of sampling points in height dimension
 *   used to compute the output.
 * - width_sample_num :specifying the number of sampling points in width dimension
 *   used to compute the output.
 */

class RoiAlign : public BuiltinOp {
 public:
  RoiAlign(Graph* graph, int32_t output_height, int32_t output_width,
            float height_ratio, float width_ratio, int32_t height_sample_num,
            int32_t width_sample_num, DataLayout input_layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  int32_t output_height_;
  int32_t output_width_;
  float height_ratio_;
  float width_ratio_;
  int32_t height_sample_num_;
  int32_t width_sample_num_;
};

using ROI_Align = RoiAlign;

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ROI_ALIGN_H_ */
