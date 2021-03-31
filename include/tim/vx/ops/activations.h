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

#define DECLARE_NO_PARAMETER_ACTIVATION(NAME) \
  class NAME : public Operation {             \
   public:                                    \
    NAME(Graph* graph);                       \
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

 protected:
  int axis_;
};

class LeakyRelu : public Operation {
 public:
  LeakyRelu(Graph* graph, float alpha);

 protected:
  float alpha_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_ACTIVATIONS_H_ */
