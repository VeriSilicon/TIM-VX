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
#include "tim/vx/ops/activations.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_NO_PARAMETER_ACTIVATION(NAME, VSI_OP_CODE)               \
  NAME::NAME(Graph* graph) : Operation(graph, VSI_OP_CODE) {}           \
  std::shared_ptr<Operation> NAME::Clone(std::shared_ptr<Graph>& graph) \
      const {                                                           \
    return graph->CreateOperation<NAME>();                              \
  }

DEFINE_NO_PARAMETER_ACTIVATION(Relu, VSI_NN_OP_RELU)
DEFINE_NO_PARAMETER_ACTIVATION(Relu1, VSI_NN_OP_RELU1)
DEFINE_NO_PARAMETER_ACTIVATION(Relu6, VSI_NN_OP_RELU6)
DEFINE_NO_PARAMETER_ACTIVATION(Elu, VSI_NN_OP_ELU)
DEFINE_NO_PARAMETER_ACTIVATION(Sigmoid, VSI_NN_OP_SIGMOID)
DEFINE_NO_PARAMETER_ACTIVATION(Mish, VSI_NN_OP_MISH)
DEFINE_NO_PARAMETER_ACTIVATION(HardSigmoid, VSI_NN_OP_HARD_SIGMOID)
DEFINE_NO_PARAMETER_ACTIVATION(SoftRelu, VSI_NN_OP_SOFTRELU)


#undef DEFINE_NO_PARAMETER_ACTIVATION

HardSwish::HardSwish(Graph* graph) : Operation(graph, VSI_NN_OP_SWISH) {
  this->impl()->node()->nn_param.swish.type = VSI_NN_HSWISH;
  this->impl()->node()->nn_param.swish.beta = 1.0f;
}

std::shared_ptr<Operation> HardSwish::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<HardSwish>();
}

Prelu::Prelu(Graph* graph, int axis)
    : Operation(graph, VSI_NN_OP_PRELU), axis_(axis) {
  this->impl()->node()->nn_param.prelu.axis = axis_;
}

std::shared_ptr<Operation> Prelu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Prelu>(this->axis_);
}

Tanh::Tanh(Graph* graph) : Operation(graph, VSI_NN_OP_TANH) {
  this->impl()->node()->nn_param.tanh.scale_a = 1.0;
  this->impl()->node()->nn_param.tanh.scale_b = 1.0;
}

std::shared_ptr<Operation> Tanh::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Tanh>();
}

LeakyRelu::LeakyRelu(Graph* graph, float alpha)
    : Operation(graph, VSI_NN_OP_LEAKY_RELU), alpha_(alpha) {
  this->impl()->node()->nn_param.activation.leaky_ratio = alpha_;
}

std::shared_ptr<Operation> LeakyRelu::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<LeakyRelu>(this->alpha_);
}

Linear::Linear(Graph* graph, float a, float b)
    : Operation(graph, VSI_NN_OP_LINEAR), a_(a), b_(b) {
  this->impl()->node()->nn_param.linear.a = a_;
  this->impl()->node()->nn_param.linear.b = b_;
}

std::shared_ptr<Operation> Linear::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Linear>(this->a_, this->b_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
