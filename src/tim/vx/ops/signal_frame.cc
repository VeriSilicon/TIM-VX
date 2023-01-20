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
#include "tim/vx/ops/signal_frame.h"
#include "vsi_nn_pub.h"
#include "builtin_op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

SignalFrame::SignalFrame(Graph* graph, uint32_t window_length, uint32_t step, uint32_t pad_end,
    uint32_t axis)
    : BuiltinOp(graph, VSI_NN_OP_SIGNAL_FRAME),
      window_length_(window_length),
      step_(step),
      pad_end_(pad_end),
      axis_(axis) {
  this->impl()->node()->nn_param.signalframe.window_length = window_length_;
  this->impl()->node()->nn_param.signalframe.step = step_;
  this->impl()->node()->nn_param.signalframe.pad_end = pad_end_;
  this->impl()->node()->nn_param.signalframe.axis = axis_;
}

std::shared_ptr<Operation> SignalFrame::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<SignalFrame>(
      this->window_length_, this->step_, this->pad_end_, this->axis_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
