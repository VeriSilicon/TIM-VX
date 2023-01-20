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
#ifndef TIM_VX_OPS_ACTIVATIONS_H_
#define TIM_VX_OPS_ACTIVATIONS_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Activation
 *
 * Activation functions:
 *
 * ```
 *   Relu(x)                : max(0, x)
 *
 *   Relu1(x)               : -1 if x <= -1; x if -1 < x < 1; 1 if x >= 1
 *
 *   Relu6(x)               : 0 if x <= 0; x if 0 < x < 6; 6 if x >= 6
 *
 *   Elu(x)                 : x if x >= 0 else alpha*(e^x - 1)
 *
 *   Tanh(x)                : (1 - e^{-2x})/(1 + e^{-2x})
 *
 *   Sigmoid(x)             : 1/(1 + e^{-x})
 *
 *   Swish(x)               : x * sigmoid(x)
 *
 *   HardSwish(x)           : 0 if x <= -3; x(x + 3)/6 if -3 < x < 3; x if x >= 3
 *
 *   HardSigmoid(x)         : min(max(alpha*x + beta, 0), 1)
 *
 *   SoftRelu(x)            : log(1 + e^x). Also known as SoftPlus.
 *
 *   Mish(x)                : x * tanh(softrelu(x))
 *
 *   LeakyRelu(x)           : alpha * x if x <= 0; x if x > 0. alpha is a scalar.
 *
 *   Prelu(x)               : alpha * x if x <= 0; x if x > 0. alpha is a tensor.
 *    - axis                : describes the axis of the inputs when coerced to 2D.
 *
 *   Linear(x, a, b)        : a*x + b.
 *
 *   Gelu(x)                : x * P(X <= x), where P(x) ~ N(0, 1). https://tensorflow.google.cn/api_docs/python/tf/nn/gelu
 *
 *   Selu(x, alpha, gamma)  : gamma * x if(x>=0), gamma * alpha * (exp(x)-1) x<0
 *
 *   Celu(x, alpha)         : x if x >= 0; alpha * (exp(x/alpha) - 1)
 * ```
 */

#define DECLARE_NO_PARAMETER_ACTIVATION(NAME)          \
  class NAME : public BuiltinOp {                    \
   public:                                             \
    NAME(Graph* graph);                                \
    std::shared_ptr<Operation> Clone(                  \
        std::shared_ptr<Graph>& graph) const override; \
  };

DECLARE_NO_PARAMETER_ACTIVATION(Relu)
DECLARE_NO_PARAMETER_ACTIVATION(Relu1)
DECLARE_NO_PARAMETER_ACTIVATION(Relu6)
DECLARE_NO_PARAMETER_ACTIVATION(Tanh)
DECLARE_NO_PARAMETER_ACTIVATION(Sigmoid)
DECLARE_NO_PARAMETER_ACTIVATION(Swish)
DECLARE_NO_PARAMETER_ACTIVATION(HardSwish)
DECLARE_NO_PARAMETER_ACTIVATION(Mish)
DECLARE_NO_PARAMETER_ACTIVATION(SoftRelu)
DECLARE_NO_PARAMETER_ACTIVATION(Sign)
DECLARE_NO_PARAMETER_ACTIVATION(SoftSign)

#undef DEFINE_NO_PARAMETER_ACTIVATION

class Elu : public BuiltinOp {
 public:
  Elu(Graph* graph);
  Elu(Graph* graph, float alpha);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
};

class Prelu : public BuiltinOp {
 public:
  Prelu(Graph* graph, int axis);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  int axis_;
};

class HardSigmoid : public BuiltinOp {
 public:
  HardSigmoid(Graph* graph, float alpha, float beta);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
  float beta_;
};

class LeakyRelu : public BuiltinOp {
 public:
  LeakyRelu(Graph* graph, float alpha);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
};

class Linear : public BuiltinOp {
 public:
  Linear(Graph* graph, float a, float b = 0.0);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float a_;
  float b_;
};

class Gelu : public BuiltinOp {
 public:
  /****************************************************************************
  *Non-approximate calculations will also have errors when the data type is
  *fp32, it is recommended to use the approximate option.
  ****************************************************************************/
  explicit Gelu(Graph* graph, bool approximate = true);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;
};

class Selu : public BuiltinOp {
 public:
  Selu(Graph* graph, float alpha = 1.67326, float gamma = 1.0507);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
  float gamma_;
};

class Celu : public BuiltinOp {
 public:
  Celu(Graph* graph, float alpha);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ACTIVATIONS_H_ */
