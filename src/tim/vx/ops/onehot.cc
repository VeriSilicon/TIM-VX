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
#include "tim/vx/ops/onehot.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
OneHot::OneHot(Graph* graph, int32_t depth, float on_value, float off_value,
               int32_t axis)
    : BuiltinOp(graph, VSI_NN_OP_ONE_HOT),
      depth_(depth),
      on_value_(on_value),
      off_value_(off_value),
      axis_(axis) {
  this->impl()->node()->nn_param.one_hot.depth = depth_;
  this->impl()->node()->nn_param.one_hot.on_value = on_value_;
  this->impl()->node()->nn_param.one_hot.off_value = off_value_;
  this->impl()->node()->nn_param.one_hot.axis = axis_;
}

std::shared_ptr<Operation> OneHot::Clone(std::shared_ptr<Graph>& graph) const {
    return graph->CreateOperation<OneHot>(this->depth_, this->on_value_,
                                          this->off_value_, this->axis_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim