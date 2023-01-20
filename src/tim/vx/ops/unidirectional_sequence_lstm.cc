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
#include "tim/vx/ops/unidirectional_sequence_lstm.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

vsi_nn_activation_e downcast_act_type(UnidirectionalSequenceLstm::ActivationType act) {
    switch (act) {
        case UnidirectionalSequenceLstm::ActivationType::kRELU:
            return VSI_NN_LSTMUNIT_ACT_RELU;
        case UnidirectionalSequenceLstm::ActivationType::kRELU6:
            return VSI_NN_LSTMUNIT_ACT_RELU6;
        case UnidirectionalSequenceLstm::ActivationType::kTANH:
            return VSI_NN_LSTMUNIT_ACT_TANH;
        case UnidirectionalSequenceLstm::ActivationType::kSIGMOID:
            return VSI_NN_LSTMUNIT_ACT_SIGMOID;
        case UnidirectionalSequenceLstm::ActivationType::kHARDSIGMOID:
            return VSI_NN_LSTMUNIT_ACT_HARD_SIGMOID;
        default: {
            VSILOGW("Not supported activition type for LSTM = %d", static_cast<int32_t>(act));
            return VSI_NN_ACT_NONE;
        }
    }
}

UnidirectionalSequenceLstm::UnidirectionalSequenceLstm(
    Graph* graph, float cell_clip, float proj_clip, ActivationType act_type,
    float forget_bias, bool time_major, ActivationType recurrent_act_type,
    bool return_sequences)
    : BuiltinOp(graph, VSI_NN_OP_LSTM_OVXLIB, LSTM_INPUT_CNT, LSTM_OUTPUT_CNT),
      act_type_(act_type),
      recurrent_act_type_(recurrent_act_type) {
  this->impl()->node()->nn_param.lstm_ovxlib.cell_clip = cell_clip;
  this->impl()->node()->nn_param.lstm_ovxlib.proj_clip = proj_clip;
  this->impl()->node()->nn_param.lstm_ovxlib.time_major = time_major;
  this->impl()->node()->nn_param.lstm_ovxlib.activation =
      downcast_act_type(act_type);
  this->impl()->node()->nn_param.lstm_ovxlib.forget_bias = forget_bias;
  this->impl()->node()->nn_param.lstm_ovxlib.recurrent_activation =
      downcast_act_type(recurrent_act_type);
  this->impl()->node()->nn_param.lstm_ovxlib.return_sequences =
      return_sequences;
}

std::shared_ptr<Operation> UnidirectionalSequenceLstm::Clone(std::shared_ptr<Graph>& graph) const {
    auto cloned_op =
        graph->CreateOperation<tim::vx::ops::UnidirectionalSequenceLstm>(
            this->impl()->node()->nn_param.lstm_ovxlib.cell_clip,
            this->impl()->node()->nn_param.lstm_ovxlib.proj_clip,
            act_type_,
            this->impl()->node()->nn_param.lstm_ovxlib.forget_bias,
            this->impl()->node()->nn_param.lstm_ovxlib.time_major,
            recurrent_act_type_,
            this->impl()->node()->nn_param.lstm_ovxlib.return_sequences);
    return cloned_op;
}

}
}  // namespace vx
}  // namespace tim
