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
#include "tim/vx/ops/grucell.h"
#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
GRUCell::GRUCell(Graph* graph, uint32_t num_units, ActivationType activation,
                 ActivationType recurrent_activation, bool reset_after)
    : BuiltinOp(graph, VSI_NN_OP_GRUCELL),
      num_units_(num_units),
      activation_(activation),
      recurrent_activation_(recurrent_activation),
      reset_after_(reset_after) {
  this->impl()->node()->nn_param.grucell.num_units = num_units;
  this->impl()->node()->nn_param.grucell.activation = activation;
  this->impl()->node()->nn_param.grucell.recurrent_activation = recurrent_activation;
  this->impl()->node()->nn_param.grucell.reset_after = TranslateToVsibool(reset_after);
}

std::shared_ptr<Operation> GRUCell::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<GRUCell>(this->num_units_, this->activation_,
                                         this->recurrent_activation_,
                                         this->reset_after_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim