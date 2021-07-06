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
#ifndef TIM_VX_OPS_SPATIAL_TRANSFORMER_H_
#define TIM_VX_OPS_SPATIAL_TRANSFORMER_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Spatial Transformer
 *
 * 'Spatial Transformer Networks', Jaderberg et. al,
 *  (https://arxiv.org/abs/1506.02025)
 * 
 * - theta : Affine transform tensor of shape (B, 6). Permits cropping,
            translation and isotropic scaling. Initialize to identity matrix.
            It is the output of the localization network.
 */

class SpatialTransformer : public Operation {
 public:
  SpatialTransformer(Graph* graph, uint32_t output_h, uint32_t output_w,
    bool has_theta_1_1, bool has_theta_1_2, bool has_theta_1_3,
    bool has_theta_2_1, bool has_theta_2_2, bool has_theta_2_3,
    float theta_1_1, float theta_1_2, float theta_1_3,
    float theta_2_1, float theta_2_2, float theta_2_3);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  const uint32_t output_h_;
  const uint32_t output_w_;
  bool has_theta_1_1_;
  bool has_theta_1_2_;
  bool has_theta_1_3_;
  bool has_theta_2_1_;
  bool has_theta_2_2_;
  bool has_theta_2_3_;
  float theta_1_1_;
  float theta_1_2_;
  float theta_1_3_;
  float theta_2_1_;
  float theta_2_2_;
  float theta_2_3_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_SPATIAL_TRANSFORMER_H_ */