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

class BidirectionalSequenceRnnExtImpl : public OpImpl {
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

  BidirectionalSequenceRnnExtImpl(Graph* graph, tim::vx::ops::BidirectionalSequenceRnn::ActivationType act_type,
              DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, layout),
      act_type_(act_type) {
    
  }

  ~BidirectionalSequenceRnnExtImpl() {}

  BidirectionalSequenceRnnExtImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;

    if (this->input_tensor_index == RNN_EXT_INPUT_CNT - 1) {
      tim::vx::DataType datatype = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetDataType();
      uint32_t input_size = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetShape()[0];
      uint32_t num_units = in_tensors_[RNN_EXT_INPUT_WEIGHT_I]->GetShape()[1];
      uint32_t batch_size = in_tensors_[RNN_EXT_INPUT_INPUT]->GetShape()[1];
      uint32_t seq_length = in_tensors_[RNN_EXT_INPUT_INPUT]->GetShape()[2];
 

      // Get all tensor
      tim::vx::ShapeType input_weight_i_shape = {input_size, num_units, 1};
      tim::vx::ShapeType input_weight_h_shape = {num_units, num_units, 1};
      tim::vx::ShapeType input_reshape_weight_i_shape = {input_size, num_units};
      tim::vx::ShapeType input_reshape_weight_h_shape = {num_units, num_units};
      tim::vx::ShapeType input_bias_shape = {2*num_units, 1};
      tim::vx::ShapeType input_reshape_bias_shape = {2*num_units};
      tim::vx::ShapeType input_reshape_split_bias_shape = {num_units};
      tim::vx::ShapeType input_hstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType input_reshape_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType output_shape = {num_units, batch_size, seq_length};
      tim::vx::ShapeType output_reshape_shape = {num_units, batch_size, 1, seq_length};
      tim::vx::ShapeType output_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType output_reshape_hstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType ext_output_shape = {num_units, batch_size, 2, seq_length};
      tim::vx::ShapeType ext_output_hstate_shape = {num_units, batch_size, 2};

      

      tim::vx::TensorSpec input_weight_i_spec(datatype, input_weight_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_weight_h_spec(datatype, input_weight_h_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_i_spec(datatype, input_reshape_weight_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_h_spec(datatype, input_reshape_weight_h_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_bias_spec(datatype, input_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_bias_spec(datatype, input_reshape_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                          
      tim::vx::TensorSpec input_reshape_split_bias_spec(datatype, input_reshape_split_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);     
      tim::vx::TensorSpec input_hstate_spec(datatype, input_hstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);         
      tim::vx::TensorSpec input_reshape_hstate_spec(datatype, input_reshape_hstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                                                                         
      tim::vx::TensorSpec output_spec(datatype, output_shape,
                                   tim::vx::TensorAttribute::TRANSIENT); 
      tim::vx::TensorSpec output_reshape_spec(datatype, output_reshape_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec output_hstate_spec(datatype, output_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec output_reshape_hstate_spec(datatype, output_reshape_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);

      auto input_fw_weight_i_tensor = graph_->CreateTensor(input_weight_i_spec);
      auto input_fw_weight_h_tensor = graph_->CreateTensor(input_weight_h_spec);
      auto input_fw_reshape_weight_i_tensor = graph_->CreateTensor(input_reshape_weight_i_spec);
      auto input_fw_reshape_weight_h_tensor = graph_->CreateTensor(input_reshape_weight_h_spec);
      auto input_fw_bias_tensor = graph_->CreateTensor(input_bias_spec);
      auto input_fw_reshape_bias_tensor = graph_->CreateTensor(input_reshape_bias_spec);
      auto input_fw_reshape_split_bias_i_tensor = graph_->CreateTensor(input_reshape_split_bias_spec);
      auto input_fw_reshape_split_bias_h_tensor = graph_->CreateTensor(input_reshape_split_bias_spec);
      auto input_fw_hstate_tensor = graph_->CreateTensor(input_hstate_spec);
      auto input_fw_reshape_hstate_tensor = graph_->CreateTensor(input_reshape_hstate_spec);
      auto output_fw_tensor = graph_->CreateTensor(output_spec);
      auto output_fw_reshape_tensor = graph_->CreateTensor(output_reshape_spec);
      auto output_fw_hstate_tensor = graph_->CreateTensor(output_hstate_spec);
      auto output_fw_reshape_hstate_tensor = graph_->CreateTensor(output_reshape_hstate_spec);

      auto input_bw_weight_i_tensor = graph_->CreateTensor(input_weight_i_spec);
      auto input_bw_weight_h_tensor = graph_->CreateTensor(input_weight_h_spec);
      auto input_bw_reshape_weight_i_tensor = graph_->CreateTensor(input_reshape_weight_i_spec);
      auto input_bw_reshape_weight_h_tensor = graph_->CreateTensor(input_reshape_weight_h_spec);
      auto input_bw_bias_tensor = graph_->CreateTensor(input_bias_spec);
      auto input_bw_reshape_bias_tensor = graph_->CreateTensor(input_reshape_bias_spec);
      auto input_bw_reshape_split_bias_i_tensor = graph_->CreateTensor(input_reshape_split_bias_spec);
      auto input_bw_reshape_split_bias_h_tensor = graph_->CreateTensor(input_reshape_split_bias_spec);
      auto input_bw_hstate_tensor = graph_->CreateTensor(input_hstate_spec);
      auto input_bw_reshape_hstate_tensor = graph_->CreateTensor(input_reshape_hstate_spec);
      auto output_bw_tensor = graph_->CreateTensor(output_spec);
      auto output_bw_reshape_tensor = graph_->CreateTensor(output_reshape_spec);
      auto output_bw_hstate_tensor = graph_->CreateTensor(output_hstate_spec);
      auto output_bw_reshape_hstate_tensor = graph_->CreateTensor(output_reshape_hstate_spec);

      std::vector<uint32_t> slices_directions = {1, 1};
      split_weight_ = graph_->CreateOperation<tim::vx::ops::Split>(2, slices_directions);
      reshape_fw_weight_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_i_shape);
      reshape_bw_weight_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_i_shape);
      
      split_recurrent_ = graph_->CreateOperation<tim::vx::ops::Split>(2, slices_directions);
      reshape_fw_recurrent_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_h_shape);
      reshape_bw_recurrent_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_h_shape);

      split_bias_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_directions);
      reshape_fw_bias_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_bias_shape);
      reshape_bw_bias_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_bias_shape);
      std::vector<uint32_t> slices_units = {num_units, num_units};
      split_reshape_fw_bias = graph_->CreateOperation<tim::vx::ops::Split>(0, slices_units);
      split_reshape_bw_bias = graph_->CreateOperation<tim::vx::ops::Split>(0, slices_units);

      split_hstate_ = graph_->CreateOperation<tim::vx::ops::Split>(2, slices_directions);
      reshape_fw_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_hstate_shape);
      reshape_bw_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_hstate_shape);
      
      rnn_ = graph_->CreateOperation<tim::vx::ops::BidirectionalSequenceRnn>(act_type_, true, false);
      
      
      reshape_fw_out_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_shape);
      reshape_fw_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_hstate_shape);
      reshape_bw_out_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_shape);
      reshape_bw_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_hstate_shape);
      concat_output_ = graph_->CreateOperation<tim::vx::ops::Concat>(2, 2);
      concat_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Concat>(2, 2);

      
      split_weight_->BindInputs({in_tensors_[RNN_EXT_INPUT_WEIGHT_I]}).
      BindOutputs({input_fw_weight_i_tensor, input_bw_weight_i_tensor});
      reshape_fw_weight_->BindInputs({input_fw_weight_i_tensor}).
      BindOutputs({input_fw_reshape_weight_i_tensor});
      reshape_bw_weight_->BindInputs({input_bw_weight_i_tensor}).
      BindOutputs({input_bw_reshape_weight_i_tensor});

      split_recurrent_->BindInputs({in_tensors_[RNN_EXT_INPUT_WEIGHT_H]}).
      BindOutputs({input_fw_weight_h_tensor, input_bw_weight_h_tensor});
      reshape_fw_recurrent_->BindInputs({input_fw_weight_h_tensor}).
      BindOutputs({input_fw_reshape_weight_h_tensor});
      reshape_bw_recurrent_->BindInputs({input_bw_weight_h_tensor}).
      BindOutputs({input_bw_reshape_weight_h_tensor});

      split_bias_->BindInputs({in_tensors_[RNN_EXT_INPUT_BIAS]}).
      BindOutputs({input_fw_bias_tensor, input_bw_bias_tensor});
      reshape_fw_bias_->BindInputs({input_fw_bias_tensor}).
      BindOutputs({input_fw_reshape_bias_tensor});
      reshape_bw_bias_->BindInputs({input_bw_bias_tensor}).
      BindOutputs({input_bw_reshape_bias_tensor});
      split_reshape_fw_bias->BindInputs({input_fw_reshape_bias_tensor}).
      BindOutputs({input_fw_reshape_split_bias_i_tensor, input_fw_reshape_split_bias_h_tensor});
      split_reshape_bw_bias->BindInputs({input_bw_reshape_bias_tensor}).
      BindOutputs({input_bw_reshape_split_bias_i_tensor, input_bw_reshape_split_bias_h_tensor});

      split_hstate_->BindInputs({in_tensors_[RNN_EXT_INPUT_H_STATE]}).
      BindOutputs({input_fw_hstate_tensor, input_bw_hstate_tensor});
      reshape_fw_hstate_->BindInputs({input_fw_hstate_tensor}).
      BindOutputs({input_fw_reshape_hstate_tensor});
      reshape_bw_hstate_->BindInputs({input_bw_hstate_tensor}).
      BindOutputs({input_bw_reshape_hstate_tensor});

      


      rnn_->BindInputs({in_tensors_[RNN_EXT_INPUT_INPUT], input_fw_reshape_weight_i_tensor, input_fw_reshape_weight_h_tensor, input_fw_reshape_split_bias_i_tensor, input_fw_reshape_split_bias_h_tensor, input_fw_reshape_hstate_tensor,
                                                          input_bw_reshape_weight_i_tensor, input_bw_reshape_weight_h_tensor, input_bw_reshape_split_bias_i_tensor, input_bw_reshape_split_bias_h_tensor, input_bw_reshape_hstate_tensor});
      rnn_->BindOutputs({output_fw_hstate_tensor, output_bw_hstate_tensor,
                        output_fw_tensor, output_bw_tensor});

      reshape_fw_out_hstate_->BindInputs({output_fw_hstate_tensor}). 
      BindOutputs({output_fw_reshape_hstate_tensor});
      reshape_fw_out_->BindInputs({output_fw_tensor}). 
      BindOutputs({output_fw_reshape_tensor});
      reshape_bw_out_hstate_->BindInputs({output_bw_hstate_tensor}). 
      BindOutputs({output_bw_reshape_hstate_tensor});
      reshape_bw_out_->BindInputs({output_bw_tensor}). 
      BindOutputs({output_bw_reshape_tensor});

      concat_out_hstate_->BindInputs({output_fw_reshape_hstate_tensor, output_bw_reshape_hstate_tensor});
      concat_output_->BindInputs({output_fw_reshape_tensor, output_bw_reshape_tensor});

    }
    this->input_tensor_index++;
    return *this;
  }

  BidirectionalSequenceRnnExtImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;

    if (this->output_tensor_index == RNN_EXT_OUT_CNT - 1) {
      concat_output_->BindOutput(out_tensors_[RNN_EXT_OUTPUT_OUTPUT]);
      concat_out_hstate_->BindOutput(out_tensors_[RNN_EXT_OUTPUT_H_STATE]);
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
  tim::vx::ops::BidirectionalSequenceRnn::ActivationType act_type_;
  
  std::shared_ptr<tim::vx::Operation> split_weight_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_weight_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_weight_;
  
  std::shared_ptr<tim::vx::Operation> split_recurrent_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_recurrent_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_recurrent_;

  std::shared_ptr<tim::vx::Operation> split_bias_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_bias_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_bias_;
  std::shared_ptr<tim::vx::Operation> split_reshape_fw_bias;
  std::shared_ptr<tim::vx::Operation> split_reshape_bw_bias;

  std::shared_ptr<tim::vx::Operation> split_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_hstate_;
  
  std::shared_ptr<tim::vx::Operation> rnn_;
  
  std::shared_ptr<tim::vx::Operation> reshape_fw_out_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_out_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_out_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_out_hstate_;
  std::shared_ptr<tim::vx::Operation> concat_output_;
  std::shared_ptr<tim::vx::Operation> concat_out_hstate_;

  std::array<std::shared_ptr<tim::vx::Tensor>, RNN_EXT_INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, RNN_EXT_OUT_CNT> out_tensors_;
};

BidirectionalSequenceRnnExt::BidirectionalSequenceRnnExt(Graph* graph, tim::vx::ops::BidirectionalSequenceRnn::ActivationType act_type)
    : act_type_(act_type) {
  impl_ = std::make_unique<BidirectionalSequenceRnnExtImpl>(graph, act_type, DataLayout::ANY);
}

std::shared_ptr<Operation> BidirectionalSequenceRnnExt::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<BidirectionalSequenceRnnExt>(this->act_type_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
