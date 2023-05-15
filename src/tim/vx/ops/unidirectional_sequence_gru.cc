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
#include "tim/vx/ops/unidirectional_sequence_gru.h"
#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
UnidirectionalSequenceGRU::UnidirectionalSequenceGRU(
    Graph* graph, uint32_t num_units, ActivationType activation,
    ActivationType recurrent_activation, bool reset_after,
    bool return_sequences, bool time_major)
    : BuiltinOp(graph, VSI_NN_OP_GRU),
      num_units_(num_units),
      activation_(activation),
      recurrent_activation_(recurrent_activation),
      reset_after_(reset_after),
      return_sequences_(return_sequences),
      time_major_(time_major) {
  this->impl()->node()->nn_param.gru.num_units = num_units;
  this->impl()->node()->nn_param.gru.activation = activation;
  this->impl()->node()->nn_param.gru.recurrent_activation = recurrent_activation;
  this->impl()->node()->nn_param.gru.reset_after = TranslateToVsibool(reset_after);
  this->impl()->node()->nn_param.gru.return_sequences = TranslateToVsibool(return_sequences);
  this->impl()->node()->nn_param.gru.time_major = TranslateToVsibool(time_major);
}

std::shared_ptr<Operation> UnidirectionalSequenceGRU::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<UnidirectionalSequenceGRU>(
      this->num_units_, this->activation_, this->recurrent_activation_,
      this->reset_after_, this->return_sequences_, this->time_major_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim