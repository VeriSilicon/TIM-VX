/****************************************************************************
*
*    Copyright (c) 2022 Vivante Corporation
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
#include "tim/vx/ops/bidirectional_sequence_rnn.h"
#include "direct_map_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
vsi_nn_activation_e downcast_act_type(
    BidirectionalSequenceRNN::ActivationType act) {
  switch (act) {
    case BidirectionalSequenceRNN::ActivationType::kRELU:
      return VSI_NN_ACT_RELU;
    case BidirectionalSequenceRNN::ActivationType::kRELU1:
      return VSI_NN_ACT_RELU1;
    case BidirectionalSequenceRNN::ActivationType::kRELU6:
      return VSI_NN_ACT_RELU6;
    case BidirectionalSequenceRNN::ActivationType::kTANH:
      return VSI_NN_ACT_TANH;
    case BidirectionalSequenceRNN::ActivationType::kSIGMOID:
      return VSI_NN_ACT_SIGMOID;
    case BidirectionalSequenceRNN::ActivationType::kHARDSIGMOID:
      return VSI_NN_ACT_HARD_SIGMOID;
    default: {
      VSILOGW("Not supported activition type for RNN = %d",
              static_cast<int32_t>(act));
      return VSI_NN_ACT_NONE;
    }
  }
}

BidirectionalSequenceRNN::BidirectionalSequenceRNN(Graph* graph,
                                                   vsi_bool time_major,
                                                   vsi_bool merge_outputs,
                                                   ActivationType act_type)
    : DirectMapOp(graph, VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_RNN),
      activation_(act_type) {
  this->impl()->node()->nn_param.bidirectional_sequence_rnn.time_major =
      time_major;
  this->impl()->node()->nn_param.bidirectional_sequence_rnn.merge_outputs =
      merge_outputs;
  this->impl()->node()->nn_param.bidirectional_sequence_rnn.activation =
      downcast_act_type(act_type);
}

std::shared_ptr<Operation> BidirectionalSequenceRNN::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<BidirectionalSequenceRNN>(this->activation_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim