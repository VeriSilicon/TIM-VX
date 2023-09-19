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
#include "tim/vx/ops/bidirectional_sequence_lstm.h"
#include "tim/vx/ops/unidirectional_sequence_lstm.h"
#include "tim/vx/ops/reverse.h"
#include "vsi_nn_pub.h"
#include "op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

class BidirectionalSequenceLstmImpl : public OpImpl {
 public:
  enum {
    BI_LSTM_INPUT_INPUT = 0,

    BI_LSTM_FW_INPUT_WEIGHT_I2I = 1,
    BI_LSTM_FW_INPUT_WEIGHT_I2F = 2,
    BI_LSTM_FW_INPUT_WEIGHT_I2C = 3,
    BI_LSTM_FW_INPUT_WEIGHT_I2O = 4,

    BI_LSTM_FW_INPUT_WEIGHT_R2I = 5,
    BI_LSTM_FW_INPUT_WEIGHT_R2F = 6,
    BI_LSTM_FW_INPUT_WEIGHT_R2C = 7,
    BI_LSTM_FW_INPUT_WEIGHT_R2O = 8,

    BI_LSTM_FW_INPUT_WEIGHT_C2I = 9,
    BI_LSTM_FW_INPUT_WEIGHT_C2F = 10,
    BI_LSTM_FW_INPUT_WEIGHT_C2O = 11,

    BI_LSTM_FW_INPUT_BIAS_I = 12,
    BI_LSTM_FW_INPUT_BIAS_F = 13,
    BI_LSTM_FW_INPUT_BIAS_C = 14,
    BI_LSTM_FW_INPUT_BIAS_O = 15,

    BI_LSTM_FW_INPUT_WEIGHT_PROJ = 16,
    BI_LSTM_FW_INPUT_BIAS_PROJ = 17,

    BI_LSTM_BW_INPUT_WEIGHT_I2I = 18,
    BI_LSTM_BW_INPUT_WEIGHT_I2F = 19,
    BI_LSTM_BW_INPUT_WEIGHT_I2C = 20,
    BI_LSTM_BW_INPUT_WEIGHT_I2O = 21,

    BI_LSTM_BW_INPUT_WEIGHT_R2I = 22,
    BI_LSTM_BW_INPUT_WEIGHT_R2F = 23,
    BI_LSTM_BW_INPUT_WEIGHT_R2C = 24,
    BI_LSTM_BW_INPUT_WEIGHT_R2O = 25,

    BI_LSTM_BW_INPUT_WEIGHT_C2I = 26,
    BI_LSTM_BW_INPUT_WEIGHT_C2F = 27,
    BI_LSTM_BW_INPUT_WEIGHT_C2O = 28,

    BI_LSTM_BW_INPUT_BIAS_I = 29,
    BI_LSTM_BW_INPUT_BIAS_F = 30,
    BI_LSTM_BW_INPUT_BIAS_C = 31,
    BI_LSTM_BW_INPUT_BIAS_O = 32,

    BI_LSTM_BW_INPUT_WEIGHT_PROJ = 33,
    BI_LSTM_BW_INPUT_BIAS_PROJ = 34,

    BI_LSTM_FW_INPUT_H_STATE = 35,
    BI_LSTM_FW_INPUT_C_STATE = 36,

    BI_LSTM_BW_INPUT_H_STATE = 37,
    BI_LSTM_BW_INPUT_C_STATE = 38,

    BI_LSTM_AUX_INPUT = 39,

    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2I = 40,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2F = 41,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2C = 42,
    BI_LSTM_FW_AUX_INPUT_WEIGHT_I2O = 43,

    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2I = 44,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2F = 45,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2C = 46,
    BI_LSTM_BW_AUX_INPUT_WEIGHT_I2O = 47,

    BI_LSTM_FW_INPUT_LAYERNORM_I = 48,
    BI_LSTM_FW_INPUT_LAYERNORM_F = 49,
    BI_LSTM_FW_INPUT_LAYERNORM_C = 50,
    BI_LSTM_FW_INPUT_LAYERNORM_O = 51,

    BI_LSTM_BW_INPUT_LAYERNORM_I = 52,
    BI_LSTM_BW_INPUT_LAYERNORM_F = 53,
    BI_LSTM_BW_INPUT_LAYERNORM_C = 54,
    BI_LSTM_BW_INPUT_LAYERNORM_O = 55,

    INPUT_CNT,

    BI_LSTM_FW_OUTPUT_OUTPUT = 0,
    BI_LSTM_FW_OUTPUT_H_STATE = 1,
    BI_LSTM_FW_OUTPUT_C_STATE = 2,

    BI_LSTM_BW_OUTPUT_OUTPUT = 3,
    BI_LSTM_BW_OUTPUT_H_STATE = 4,
    BI_LSTM_BW_OUTPUT_C_STATE = 5,

    OUTPUT_CNT
  };

  BidirectionalSequenceLstmImpl(Graph* graph, int input_cnt, int output_cnt,
                                 float cell_clip,  float proj_clip,
                                 tim::vx::ops::UnidirectionalSequenceLstm::ActivationType act_type,
                                 float forget_bias,  bool time_major,
                                 tim::vx::ops::UnidirectionalSequenceLstm::ActivationType recurrent_act_type,
                                 bool return_sequences, DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, -1, input_cnt, output_cnt, layout) {
      lstm_forward_ = graph->CreateOperation<UnidirectionalSequenceLstm>(
        cell_clip, proj_clip, act_type, forget_bias, time_major,
        recurrent_act_type, return_sequences);
      lstm_backward_ = graph->CreateOperation<UnidirectionalSequenceLstm>(
        cell_clip, proj_clip, act_type, forget_bias, time_major,
        recurrent_act_type, return_sequences);
      reverse_input_ =  graph->CreateOperation<Reverse>(time_major ? std::vector<int32_t> ({2}) :
                                                       std::vector<int32_t> ({1}));
      reverse_output_ =  graph->CreateOperation<Reverse>(time_major ? std::vector<int32_t> ({2}) :
                                                       std::vector<int32_t> ({1}));
  }

  ~BidirectionalSequenceLstmImpl() {}

  BidirectionalSequenceLstmImpl& BindInput(
      const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;

    if (this->input_tensor_index == INPUT_CNT - 1) {
      // Get all input tensor
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_INPUT_INPUT]);
      reverse_input_->BindInput(in_tensors_[BI_LSTM_INPUT_INPUT]);
      TensorSpec bw_input_spec (in_tensors_[BI_LSTM_INPUT_INPUT]->GetSpec());
      bw_input_tensor_ = graph_->CreateTensor(bw_input_spec.AsTransientSpec());
      reverse_input_->BindOutput(bw_input_tensor_);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_H_STATE]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_C_STATE]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_I2I]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_I2F]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_I2C]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_I2O]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_R2I]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_R2F]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_R2C]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_R2O]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_C2I]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_C2F]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_C2O]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_BIAS_I]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_BIAS_F]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_BIAS_C]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_BIAS_O]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_WEIGHT_PROJ]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_BIAS_PROJ]);

      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_LAYERNORM_I]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_LAYERNORM_F]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_LAYERNORM_C]);
      lstm_forward_->BindInput(in_tensors_[BI_LSTM_FW_INPUT_LAYERNORM_O]);

      lstm_backward_->BindInput(bw_input_tensor_);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_H_STATE]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_C_STATE]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_I2I]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_I2F]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_I2C]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_I2O]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_R2I]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_R2F]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_R2C]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_R2O]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_C2I]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_C2F]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_C2O]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_BIAS_I]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_BIAS_F]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_BIAS_C]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_BIAS_O]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_WEIGHT_PROJ]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_BIAS_PROJ]);

      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_LAYERNORM_I]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_LAYERNORM_F]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_LAYERNORM_C]);
      lstm_backward_->BindInput(in_tensors_[BI_LSTM_BW_INPUT_LAYERNORM_O]);
    }
    this->input_tensor_index++;
    return *this;
  }

  BidirectionalSequenceLstmImpl& BindOutput(
      const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;

    if (this->output_tensor_index == OUTPUT_CNT - 1) {
      lstm_forward_->BindOutput(out_tensors_[BI_LSTM_FW_OUTPUT_OUTPUT]);
      lstm_forward_->BindOutput(out_tensors_[BI_LSTM_FW_OUTPUT_H_STATE]);
      lstm_forward_->BindOutput(out_tensors_[BI_LSTM_FW_OUTPUT_C_STATE]);

      bw_output_tensor_ = graph_->CreateTensor(out_tensors_[BI_LSTM_BW_OUTPUT_OUTPUT]->GetSpec());
      lstm_backward_->BindOutput(bw_output_tensor_);
      reverse_output_->BindInput(bw_output_tensor_);
      reverse_output_->BindOutput(out_tensors_[BI_LSTM_BW_OUTPUT_OUTPUT]);
      lstm_backward_->BindOutput(out_tensors_[BI_LSTM_BW_OUTPUT_H_STATE]);
      lstm_backward_->BindOutput(out_tensors_[BI_LSTM_BW_OUTPUT_C_STATE]);
    }
    this->output_tensor_index++;
    return *this;
  }

  vsi_nn_node_t* node() override { return nullptr; }

  std::vector<std::shared_ptr<Tensor>> InputsTensor() override {
    return inputs_tensor_;
  }
  std::vector<std::shared_ptr<Tensor>> OutputsTensor() override {
    return outputs_tensor_;
  }

 private:
  std::shared_ptr<tim::vx::Operation> lstm_forward_;
  std::shared_ptr<tim::vx::Operation> lstm_backward_;
  std::shared_ptr<tim::vx::Operation> reverse_input_;
  std::shared_ptr<tim::vx::Operation> reverse_output_;

  std::array<std::shared_ptr<tim::vx::Tensor>, INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, OUTPUT_CNT> out_tensors_;
  std::shared_ptr<Tensor> bw_input_tensor_;
  std::shared_ptr<Tensor> bw_output_tensor_;
};

UnidirectionalSequenceLstm::ActivationType interpreter(BidirectionalSequenceLstm::ActivationType act){
   switch (act){

    case BidirectionalSequenceLstm::ActivationType::kRELU:
        return UnidirectionalSequenceLstm::ActivationType::kRELU;
    case BidirectionalSequenceLstm::ActivationType::kRELU6:
        return UnidirectionalSequenceLstm::ActivationType::kRELU6;
    case BidirectionalSequenceLstm::ActivationType::kTANH:
        return UnidirectionalSequenceLstm::ActivationType::kTANH;
    case BidirectionalSequenceLstm::ActivationType::kSIGMOID:
        return UnidirectionalSequenceLstm::ActivationType::kSIGMOID;
    case BidirectionalSequenceLstm::ActivationType::kHARDSIGMOID:
        return UnidirectionalSequenceLstm::ActivationType::kHARDSIGMOID;
    default: {
        return UnidirectionalSequenceLstm::ActivationType::kNONE;
   }
  }
}
BidirectionalSequenceLstm::BidirectionalSequenceLstm(
    Graph* graph, float cell_clip, float proj_clip, ActivationType act_type,
    float forget_bias, bool time_major, ActivationType recurrent_act_type,
    bool return_sequences)
    : cell_clip_(cell_clip),
      proj_clip_(proj_clip),
      act_type_(act_type),
      forget_bias_(forget_bias),
      time_major_(time_major),
      recurrent_act_type_(recurrent_act_type),
      return_sequences_(return_sequences) {
  impl_ = std::make_unique<BidirectionalSequenceLstmImpl>(graph, 0, 0, cell_clip_,
                                                         proj_clip_, interpreter(act_type_),
                                                         forget_bias_,time_major_,
                                                         interpreter(recurrent_act_type_),
                                                         return_sequences_, DataLayout::ANY);
}

std::shared_ptr<Operation> BidirectionalSequenceLstm::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<BidirectionalSequenceLstm>(
      this->cell_clip_, this->proj_clip_, this->act_type_, this->forget_bias_,
      this->time_major_, this->recurrent_act_type_, this->return_sequences_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
