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
#include "tim/vx/ops/activations.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_NO_PARAMETER_ACTIVATION(NAME, VSI_OP_CODE)               \
  NAME::NAME(Graph* graph) : BuiltinOp(graph, VSI_OP_CODE) {}         \
  std::shared_ptr<Operation> NAME::Clone(std::shared_ptr<Graph>& graph) \
      const {                                                           \
    return graph->CreateOperation<NAME>();                              \
  }

DEFINE_NO_PARAMETER_ACTIVATION(Relu, VSI_NN_OP_RELU)
DEFINE_NO_PARAMETER_ACTIVATION(Relu1, VSI_NN_OP_RELU1)
DEFINE_NO_PARAMETER_ACTIVATION(Relu6, VSI_NN_OP_RELU6)
DEFINE_NO_PARAMETER_ACTIVATION(Sigmoid, VSI_NN_OP_SIGMOID)
DEFINE_NO_PARAMETER_ACTIVATION(Mish, VSI_NN_OP_MISH)
DEFINE_NO_PARAMETER_ACTIVATION(SoftRelu, VSI_NN_OP_SOFTRELU)
DEFINE_NO_PARAMETER_ACTIVATION(Sign,VSI_NN_OP_SIGN)
DEFINE_NO_PARAMETER_ACTIVATION(SoftSign,VSI_NN_OP_SOFTSIGN)

#undef DEFINE_NO_PARAMETER_ACTIVATION

Elu::Elu(Graph* graph) : Elu(graph, 1) {}

Elu::Elu(Graph* graph, float alpha)
    : BuiltinOp(graph, VSI_NN_OP_ELU), alpha_(alpha) {
  this->impl()->node()->nn_param.elu.alpha = alpha_;
}

std::shared_ptr<Operation> Elu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Elu>(this->alpha_);
}

HardSwish::HardSwish(Graph* graph) : BuiltinOp(graph, VSI_NN_OP_SWISH) {
  this->impl()->node()->nn_param.swish.type = VSI_NN_HSWISH;
  this->impl()->node()->nn_param.swish.beta = 1.0f;
}

std::shared_ptr<Operation> HardSwish::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<HardSwish>();
}

Swish::Swish(Graph* graph) : BuiltinOp(graph, VSI_NN_OP_SWISH) {
  this->impl()->node()->nn_param.swish.type = VSI_NN_SWISH;
  this->impl()->node()->nn_param.swish.beta = 1.0f;
}

std::shared_ptr<Operation> Swish::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Swish>();
}

Prelu::Prelu(Graph* graph, int axis)
    : BuiltinOp(graph, VSI_NN_OP_PRELU), axis_(axis) {
  this->impl()->node()->nn_param.prelu.axis = axis_;
}

std::shared_ptr<Operation> Prelu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Prelu>(this->axis_);
}

HardSigmoid::HardSigmoid(Graph* graph, float alpha, float beta)
    : BuiltinOp(graph, VSI_NN_OP_HARD_SIGMOID), alpha_(alpha), beta_(beta) {
  this->impl()->node()->nn_param.hard_sigmoid.alpha = alpha_;
  this->impl()->node()->nn_param.hard_sigmoid.beta = beta_;
}

std::shared_ptr<Operation> HardSigmoid::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<HardSigmoid>(this->alpha_, this->beta_);
}

Tanh::Tanh(Graph* graph) : BuiltinOp(graph, VSI_NN_OP_TANH) {
  this->impl()->node()->nn_param.tanh.scale_a = 1.0;
  this->impl()->node()->nn_param.tanh.scale_b = 1.0;
}

std::shared_ptr<Operation> Tanh::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Tanh>();
}

LeakyRelu::LeakyRelu(Graph* graph, float alpha)
    : BuiltinOp(graph, VSI_NN_OP_LEAKY_RELU), alpha_(alpha) {
  this->impl()->node()->nn_param.activation.leaky_ratio = alpha_;
}

std::shared_ptr<Operation> LeakyRelu::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<LeakyRelu>(this->alpha_);
}

Linear::Linear(Graph* graph, float a, float b)
    : BuiltinOp(graph, VSI_NN_OP_LINEAR), a_(a), b_(b) {
  this->impl()->node()->nn_param.linear.a = a_;
  this->impl()->node()->nn_param.linear.b = b_;
}

std::shared_ptr<Operation> Linear::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Linear>(this->a_, this->b_);
}

Gelu::Gelu(Graph* graph, bool approximate)
    : BuiltinOp(graph, VSI_NN_OP_GELU) {
  this->impl()->node()->nn_param.gelu.approximate = approximate;
}

std::shared_ptr<Operation> Gelu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Gelu>(
      this->impl()->node()->nn_param.gelu.approximate);
}

#ifdef _VSI_NN_OP_SELU_H
Selu::Selu(Graph* graph, float alpha, float gamma)
    : BuiltinOp(graph, VSI_NN_OP_SELU), alpha_(alpha), gamma_(gamma) {
  this->impl()->node()->nn_param.selu.alpha = alpha;
  this->impl()->node()->nn_param.selu.gamma = gamma;
}

std::shared_ptr<Operation> Selu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Selu>(this->alpha_, this->gamma_);
}
#endif

#ifdef _VSI_NN_OP_CELU_H
Celu::Celu(Graph* graph, float alpha)
    : BuiltinOp(graph, VSI_NN_OP_CELU), alpha_(alpha) {
  this->impl()->node()->nn_param.selu.alpha = alpha;
}

std::shared_ptr<Operation> Celu::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Celu>(this->alpha_);
}
#endif

}  // namespace ops
}  // namespace vx
}  // namespace tim
