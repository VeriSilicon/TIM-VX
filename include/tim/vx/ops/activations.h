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
#ifndef TIM_VX_OPS_ACTIVATIONS_H_
#define TIM_VX_OPS_ACTIVATIONS_H_
#include "tim/vx/operation.h"

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
 *   HardSwish(x)           : 0 if x <= -3; x(x + 3)/6 if -3 < x < 3; x if x >= 3
 *
 *   Mish(x)                : x if x >= 0 else alpha * x
 *
 *   HardSigmoid(x)         : min(max(alpha*x + beta, 0), 1)
 *
 *   SoftRelu(x)            : log(1 + e^x). Also known as SoftPlus.
 *
 *   LeakyRelu(x)           : alpha * x if x <= 0; x if x > 0. alpha is a scalar.
 *
 *   Prelu(x)               : alpha * x if x <= 0; x if x > 0. alpha is a tensor.
 *    - axis                : describes the axis of the inputs when coerced to 2D.
 *
 *   Linear(x, a, b)        : a*x + b.
 * ```
 */

#define DECLARE_NO_PARAMETER_ACTIVATION(NAME)          \
  class NAME : public Operation {                      \
   public:                                             \
    NAME(Graph* graph);                                \
    std::shared_ptr<Operation> Clone(                  \
        std::shared_ptr<Graph>& graph) const override; \
  };

DECLARE_NO_PARAMETER_ACTIVATION(Relu)
DECLARE_NO_PARAMETER_ACTIVATION(Relu1)
DECLARE_NO_PARAMETER_ACTIVATION(Relu6)
DECLARE_NO_PARAMETER_ACTIVATION(Elu)
DECLARE_NO_PARAMETER_ACTIVATION(Tanh)
DECLARE_NO_PARAMETER_ACTIVATION(Sigmoid)
DECLARE_NO_PARAMETER_ACTIVATION(HardSwish)
DECLARE_NO_PARAMETER_ACTIVATION(Mish)
DECLARE_NO_PARAMETER_ACTIVATION(HardSigmoid)
DECLARE_NO_PARAMETER_ACTIVATION(SoftRelu)

#undef DEFINE_NO_PARAMETER_ACTIVATION

class Prelu : public Operation {
 public:
  Prelu(Graph* graph, int axis);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  int axis_;
};

class LeakyRelu : public Operation {
 public:
  LeakyRelu(Graph* graph, float alpha);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float alpha_;
};

class Linear : public Operation {
 public:
  Linear(Graph* graph, float a, float b = 0.0);
  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  float a_;
  float b_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ACTIVATIONS_H_ */
