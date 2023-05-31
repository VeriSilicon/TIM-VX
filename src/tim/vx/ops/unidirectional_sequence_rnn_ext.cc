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
#include "tim/vx/ops.h"
#include "vsi_nn_pub.h"
#include "op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

class UnidirectionalSequenceRnnExtImpl : public OpImpl {
 public:
  enum {
    // signature
    RNN_EXT_INPUT_INPUT = 0,
    RNN_EXT_INPUT_WEIGHT_I = 1,
    RNN_EXT_INPUT_WEIGHT_H = 2,
    RNN_EXT_INPUT_BIAS = 3,
    RNN_EXT_INPUT_H_STATE = 4,
    RNN_EXT_INPUT_CNT,

    RNN_EXT_OUTPUT_H_STATE = 0,
    RNN_EXT_OUTPUT_OUTPUT = 1,
    RNN_EXT_OUT_CNT,
    // signature end
  };

  UnidirectionalSequenceRnnExtImpl(Graph* graph, tim::vx::ops::UnidirectionalSequenceRnn::ActivationType act_type,
              DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, layout),
      act_type_(act_type) {
    
  }

  ~UnidirectionalSequenceRnnExtImpl() {}

  UnidirectionalSequenceRnnExtImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;

    if (this->input_tensor_index == RNN_EXT_INPUT_CNT - 1) {
      tim::vx::DataType datatype = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetDataType();
      uint32_t input_size = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetShape()[0];
      uint32_t num_units = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetShape()[1];
      uint32_t batch_size = in_tensors_[RNN_EXT_INPUT_INPUT]->GetShape()[1];
      uint32_t seq_length = in_tensors_[RNN_EXT_INPUT_INPUT]->GetShape()[2];
 

      // Get all tensor
      tim::vx::ShapeType input_weight_i_shape = {input_size, num_units};
      tim::vx::ShapeType input_weight_h_shape = {num_units, num_units};
      tim::vx::ShapeType input_bias_shape = {2*num_units};
      tim::vx::ShapeType input_bias_i_shape = {num_units};
      tim::vx::ShapeType input_bias_h_shape = {num_units};
      tim::vx::ShapeType input_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType output_shape = {num_units, batch_size, seq_length};
      tim::vx::ShapeType output_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType ext_output_shape = {num_units, 1, batch_size, seq_length};
      tim::vx::ShapeType ext_output_hstate_shape = {num_units, batch_size, 1};

      

      tim::vx::TensorSpec input_weight_i_spec(datatype, input_weight_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_weight_h_spec(datatype, input_weight_h_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_bias_spec(datatype, input_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                              
      tim::vx::TensorSpec input_bias_i_spec(datatype, input_bias_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);   
      tim::vx::TensorSpec input_bias_h_spec(datatype, input_bias_h_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);    
      tim::vx::TensorSpec input_hstate_spec(datatype, input_hstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                                                                                
      tim::vx::TensorSpec output_spec(datatype, output_shape,
                                   tim::vx::TensorAttribute::TRANSIENT); 
      tim::vx::TensorSpec output_hstate_spec(datatype, output_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);

      auto input_weight_i_tensor = graph_->CreateTensor(input_weight_i_spec);
      auto input_weight_h_tensor = graph_->CreateTensor(input_weight_h_spec);
      auto input_bias_tensor = graph_->CreateTensor(input_bias_spec);
      auto input_bias_i_tensor = graph_->CreateTensor(input_bias_i_spec);
      auto input_bias_h_tensor = graph_->CreateTensor(input_bias_h_spec);
      auto input_hstate_tensor = graph_->CreateTensor(input_hstate_spec);
      auto output_tensor = graph_->CreateTensor(output_spec);
      auto output_hstate_tensor = graph_->CreateTensor(output_hstate_spec);

      reshape_weight_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_weight_i_shape);
      reshape_recurrent_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_weight_h_shape);
      reshape_bias_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_bias_shape);
      std::vector<uint32_t> slices = {num_units, num_units};
      split_ = graph_->CreateOperation<tim::vx::ops::Split>(0, slices);
      reshape_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_hstate_shape);
      rnn_ = graph_->CreateOperation<tim::vx::ops::UnidirectionalSequenceRnn>(act_type_, true);
      reshape_out_ = graph_->CreateOperation<tim::vx::ops::Reshape>(ext_output_shape);
      reshape_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(ext_output_hstate_shape);
      

      reshape_weight_->BindInput(in_tensors_[RNN_EXT_INPUT_WEIGHT_I]);
      reshape_weight_->BindOutput(input_weight_i_tensor);

      reshape_recurrent_->BindInput(in_tensors_[RNN_EXT_INPUT_WEIGHT_H]);
      reshape_recurrent_->BindOutput(input_weight_h_tensor);

      reshape_bias_->BindInput(in_tensors_[RNN_EXT_INPUT_BIAS]);
      reshape_bias_->BindOutput(input_bias_tensor);
      split_->BindInput(input_bias_tensor);
      split_->BindOutput(input_bias_i_tensor);
      split_->BindOutput(input_bias_h_tensor);

      reshape_hstate_->BindInput(in_tensors_[RNN_EXT_INPUT_H_STATE]);
      reshape_hstate_->BindOutput(input_hstate_tensor);

      rnn_->BindInputs({in_tensors_[RNN_EXT_INPUT_INPUT], input_weight_i_tensor, input_weight_h_tensor, input_bias_i_tensor, input_bias_h_tensor, input_hstate_tensor});
      rnn_->BindOutputs({output_hstate_tensor, output_tensor});

      reshape_out_->BindInput(output_tensor);
      reshape_out_hstate_->BindInput(output_hstate_tensor);

    }
    this->input_tensor_index++;
    return *this;
  }

  UnidirectionalSequenceRnnExtImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;

    if (this->output_tensor_index == RNN_EXT_OUT_CNT - 1) {
      reshape_out_->BindOutput(out_tensors_[RNN_EXT_OUTPUT_OUTPUT]);
      reshape_out_hstate_->BindOutput(out_tensors_[RNN_EXT_OUTPUT_H_STATE]);
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
  tim::vx::ops::UnidirectionalSequenceRnn::ActivationType act_type_;
  std::shared_ptr<tim::vx::Operation> reshape_weight_;
  std::shared_ptr<tim::vx::Operation> reshape_recurrent_;
  std::shared_ptr<tim::vx::Operation> reshape_bias_;
  std::shared_ptr<tim::vx::Operation> split_;
  std::shared_ptr<tim::vx::Operation> reshape_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_out_;
  std::shared_ptr<tim::vx::Operation> reshape_out_hstate_;
  std::shared_ptr<tim::vx::Operation> rnn_;

  std::array<std::shared_ptr<tim::vx::Tensor>, RNN_EXT_INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, RNN_EXT_OUT_CNT> out_tensors_;
};

UnidirectionalSequenceRnnExt::UnidirectionalSequenceRnnExt(Graph* graph, tim::vx::ops::UnidirectionalSequenceRnn::ActivationType act_type)
    : act_type_(act_type) {
  impl_ = std::make_unique<UnidirectionalSequenceRnnExtImpl>(graph, act_type, DataLayout::ANY);
}

std::shared_ptr<Operation> UnidirectionalSequenceRnnExt::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<UnidirectionalSequenceRnnExt>(this->act_type_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
