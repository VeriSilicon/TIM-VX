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
#include "tim/vx/ops/spatial_transformer.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

SpatialTransformer::SpatialTransformer(Graph* graph, uint32_t output_h, uint32_t output_w,
    bool has_theta_1_1, bool has_theta_1_2, bool has_theta_1_3,
    bool has_theta_2_1, bool has_theta_2_2, bool has_theta_2_3,
    float theta_1_1, float theta_1_2, float theta_1_3,
    float theta_2_1, float theta_2_2, float theta_2_3)
    : Operation(graph, VSI_NN_OP_SPATIAL_TRANSFORMER, 2, 1), output_h_(output_h), output_w_(output_w),
    has_theta_1_1_(has_theta_1_1), has_theta_1_2_(has_theta_1_2), has_theta_1_3_(has_theta_1_3),
    has_theta_2_1_(has_theta_2_1), has_theta_2_2_(has_theta_2_2), has_theta_2_3_(has_theta_2_3),
    theta_1_1_(theta_1_1), theta_1_2_(theta_1_2), theta_1_3_(theta_1_3),
    theta_2_1_(theta_2_1), theta_2_2_(theta_2_2), theta_2_3_(theta_2_3) {
  this->impl()->node()->nn_param.spatial_transformer.output_H = output_h_;
  this->impl()->node()->nn_param.spatial_transformer.output_W = output_w_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_1_1 = has_theta_1_1_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_1_2 = has_theta_1_2_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_1_3 = has_theta_1_3_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_2_1 = has_theta_2_1_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_2_2 = has_theta_2_2_;
  this->impl()->node()->nn_param.spatial_transformer.has_theta_2_3 = has_theta_2_3_;
  this->impl()->node()->nn_param.spatial_transformer.theta_1_1 = theta_1_1_;
  this->impl()->node()->nn_param.spatial_transformer.theta_1_2 = theta_1_2_;
  this->impl()->node()->nn_param.spatial_transformer.theta_1_3 = theta_1_3_;
  this->impl()->node()->nn_param.spatial_transformer.theta_2_1 = theta_2_1_;
  this->impl()->node()->nn_param.spatial_transformer.theta_2_2 = theta_2_2_;
  this->impl()->node()->nn_param.spatial_transformer.theta_2_3 = theta_2_3_;
}

std::shared_ptr<Operation> SpatialTransformer::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<SpatialTransformer>(
      this->output_h_, this->output_w_, this->has_theta_1_1_,
      this->has_theta_1_2_, this->has_theta_1_3_, this->has_theta_2_1_,
      this->has_theta_2_2_, this->has_theta_2_3_, this->theta_1_1_,
      this->theta_1_2_, this->theta_1_3_, this->theta_2_1_, this->theta_2_2_,
      this->theta_2_3_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim