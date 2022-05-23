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
#include "tim/vx/ops/bidirectional_sequence_lstm.h"

#include "direct_map_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

extern vsi_nn_activation_e downcast_act_type(UnidirectionalSequenceLstm::ActivationType act);

BidirectionalSequenceLstm::BidirectionalSequenceLstm(
    Graph* graph, float cell_clip, float proj_clip, ActivationType act_type,
    float forget_bias, bool time_major, ActivationType recurrent_act_type,
    bool merge_outputs)
    : DirectMapOp(graph, VSI_NN_OP_BIDIRECTIONAL_SEQUENCE_LSTM, BI_LSTM_INPUT_CNT, BI_LSTM_OUTPUT_CNT),
      act_type_(act_type),
      recurrent_act_type_(recurrent_act_type) {
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.cell_clip = cell_clip;
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.proj_clip = proj_clip;
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.time_major = time_major;
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.activation =
      downcast_act_type(act_type);
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.forget_bias = forget_bias;
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.recurrent_activation =
      downcast_act_type(recurrent_act_type);
  this->impl()->node()->nn_param.bidirectional_sequence_lstm.merge_outputs = merge_outputs;
}

std::shared_ptr<Operation> BidirectionalSequenceLstm::Clone(std::shared_ptr<Graph>& graph) const {
    auto cloned_op =
        graph->CreateOperation<tim::vx::ops::BidirectionalSequenceLstm>(
            this->impl()->node()->nn_param.bidirectional_sequence_lstm.cell_clip,
            this->impl()->node()->nn_param.bidirectional_sequence_lstm.proj_clip,
            act_type_,
            this->impl()->node()->nn_param.bidirectional_sequence_lstm.forget_bias,
            this->impl()->node()->nn_param.bidirectional_sequence_lstm.time_major,
            recurrent_act_type_,
            this->impl()->node()->nn_param.bidirectional_sequence_lstm.merge_outputs);
    return cloned_op;
}

}
}  // namespace vx
}  // namespace tim
