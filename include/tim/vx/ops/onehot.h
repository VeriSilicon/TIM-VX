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
#ifndef TIM_VX_OPERATION_ONE_HOT_H_
#define TIM_VX_OPERATION_ONE_HOT_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## OneHot
 *
 * Create a one-hot tensor.
 *
 * - depth : A scalar defining the depth of the one hot dimension.
 * - on_value : A scalar defining the value to fill in output.
 * - off_value : A scalar defining the value to fill in output.
 * - axis : The axis to fill.
 */

class OneHot : public BuiltinOp {
 public:
  OneHot(Graph* graph, int32_t depth, float on_value = 1, float off_value = 0,
         int32_t axis = 0);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  int32_t depth_;
  float on_value_;
  float off_value_;
  int32_t axis_;
};
}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif