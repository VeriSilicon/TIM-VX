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
#include "tim/vx/ops.h"
#include "vsi_nn_pub.h"
#include "op_impl.h"

#include <array>
namespace tim {
namespace vx {
namespace ops {

class BidirectionalSequenceLstmExtImpl : public OpImpl {
 public:
  enum {
    // signature
    BI_LSTM_EXT_INPUT_INPUT = 0,
    BI_LSTM_EXT_INPUT_WEIGHT_I = 1,
    BI_LSTM_EXT_INPUT_WEIGHT_R = 2,
    BI_LSTM_EXT_INPUT_BIAS = 3,
    BI_LSTM_EXT_INPUT_H_STATE = 4,
    BI_LSTM_EXT_INPUT_C_STATE = 5,
    BI_LSTM_EXT_INPUT_WEIGHT_P = 6,
    BI_LSTM_EXT_INPUT_CNT,

    BI_LSTM_EXT_OUTPUT_OUTPUT = 0,
    BI_LSTM_EXT_OUTPUT_H_STATE = 1,
    BI_LSTM_EXT_OUTPUT_C_STATE = 2,
    BI_LSTM_EXT_OUT_CNT,
    // signature end
  };

  BidirectionalSequenceLstmExtImpl(Graph* graph, float cell_clip,
              tim::vx::ops::BidirectionalSequenceLstm::ActivationType act_type,
              bool time_major,
              tim::vx::ops::BidirectionalSequenceLstm::ActivationType recurrent_act_type,
              DataLayout layout = DataLayout::ANY)
      : OpImpl(graph, layout),
      cell_clip_(cell_clip),
      act_type_(act_type),
      time_major_(time_major),
      recurrent_act_type_(recurrent_act_type) {
    
  }

  ~BidirectionalSequenceLstmExtImpl() {}

  BidirectionalSequenceLstmExtImpl& BindInput(const std::shared_ptr<Tensor>& tensor) override {
    in_tensors_[input_tensor_index] = tensor;

    if (this->input_tensor_index == BI_LSTM_EXT_INPUT_CNT - 1) {
      tim::vx::DataType datatype = in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_I]->GetDataType();
      uint32_t input_size = in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_I]->GetShape()[0];
      uint32_t num_units = in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_R]->GetShape()[0];
      uint32_t batch_size = 0;
      uint32_t seq_length = 0;
      if(time_major_)
      {
          batch_size = in_tensors_[BI_LSTM_EXT_INPUT_INPUT]->GetShape()[1];
          seq_length = in_tensors_[BI_LSTM_EXT_INPUT_INPUT]->GetShape()[2];
      }
      else
      {
          batch_size = in_tensors_[BI_LSTM_EXT_INPUT_INPUT]->GetShape()[2];
          seq_length = in_tensors_[BI_LSTM_EXT_INPUT_INPUT]->GetShape()[1];
      }
      
 

      // Get all tensor
      tim::vx::ShapeType input_weight_i_shape = {input_size, 4*num_units, 1};
      tim::vx::ShapeType input_reshape_weight_i_shape = {input_size, 4*num_units};
      tim::vx::ShapeType input_reshape_weight_i2i_shape = {input_size, num_units};
      tim::vx::ShapeType input_reshape_weight_i2o_shape = {input_size, num_units};
      tim::vx::ShapeType input_reshape_weight_i2f_shape = {input_size, num_units};
      tim::vx::ShapeType input_reshape_weight_i2c_shape = {input_size, num_units};
      tim::vx::ShapeType input_weight_r_shape = {num_units, 4*num_units, 1};
      tim::vx::ShapeType input_reshape_weight_r_shape = {num_units, 4*num_units};
      tim::vx::ShapeType input_reshape_weight_r2i_shape = {num_units, num_units};
      tim::vx::ShapeType input_reshape_weight_r2o_shape = {num_units, num_units};
      tim::vx::ShapeType input_reshape_weight_r2f_shape = {num_units, num_units};
      tim::vx::ShapeType input_reshape_weight_r2c_shape = {num_units, num_units};
      tim::vx::ShapeType input_bias_shape = {8*num_units, 1};
      tim::vx::ShapeType input_reshape_bias_shape = {8*num_units};
      tim::vx::ShapeType input_reshape_bias_i_shape = {num_units};
      tim::vx::ShapeType input_reshape_bias_r_shape = {num_units};
      tim::vx::ShapeType input_hstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType input_reshape_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType input_cstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType input_reshape_cstate_shape = {num_units, batch_size};
      tim::vx::ShapeType input_weight_p_shape = {num_units, 3*num_units, 1};
      tim::vx::ShapeType input_reshape_weight_p_shape = {num_units, 3*num_units};
      tim::vx::ShapeType input_reshape_weight_p2i_shape = {num_units, num_units};
      tim::vx::ShapeType input_reshape_weight_p2o_shape = {num_units, num_units};
      tim::vx::ShapeType input_reshape_weight_p2f_shape = {num_units, num_units};
      tim::vx::ShapeType output_shape;
      tim::vx::ShapeType output_reshape_shape;
      tim::vx::ShapeType ext_output_shape;
      if(time_major_)
      {
          output_shape = {num_units, batch_size, seq_length};
          output_reshape_shape = {num_units, batch_size, 1, seq_length};
          ext_output_shape = {num_units, batch_size, 2, seq_length};
      }
      else
      {
          output_shape = {num_units, seq_length, batch_size};
          output_reshape_shape = {num_units, 1, seq_length, batch_size};
          ext_output_shape = {num_units, 2, seq_length, batch_size};
      }
      tim::vx::ShapeType output_hstate_shape = {num_units, batch_size};
      tim::vx::ShapeType output_reshape_hstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType output_cstate_shape = {num_units, batch_size};
      tim::vx::ShapeType output_reshape_cstate_shape = {num_units, batch_size, 1};
      tim::vx::ShapeType ext_output_hstate_shape = {num_units, batch_size, 2};
      tim::vx::ShapeType ext_output_cstate_shape = {num_units, batch_size, 2};

      

      tim::vx::TensorSpec input_weight_i_spec(datatype, input_weight_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_i_spec(datatype, input_reshape_weight_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_i2i_spec(datatype, input_reshape_weight_i2i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_i2o_spec(datatype, input_reshape_weight_i2o_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);  
      tim::vx::TensorSpec input_reshape_weight_i2f_spec(datatype, input_reshape_weight_i2f_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_i2c_spec(datatype, input_reshape_weight_i2c_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);        
      tim::vx::TensorSpec input_weight_r_spec(datatype, input_weight_r_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_r_spec(datatype, input_reshape_weight_r_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_r2i_spec(datatype, input_reshape_weight_r2i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_r2o_spec(datatype, input_reshape_weight_r2o_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_r2f_spec(datatype, input_reshape_weight_r2f_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_r2c_spec(datatype, input_reshape_weight_r2c_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);  
      tim::vx::TensorSpec input_bias_spec(datatype, input_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_bias_spec(datatype, input_reshape_bias_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                              
      tim::vx::TensorSpec input_reshape_bias_i_spec(datatype, input_reshape_bias_i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);   
      tim::vx::TensorSpec input_reshape_bias_r_spec(datatype, input_reshape_bias_r_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);    
      tim::vx::TensorSpec input_hstate_spec(datatype, input_hstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);      
      tim::vx::TensorSpec input_reshape_hstate_spec(datatype, input_reshape_hstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);     
      tim::vx::TensorSpec input_cstate_spec(datatype, input_cstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_cstate_spec(datatype, input_reshape_cstate_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_weight_p_spec(datatype, input_weight_p_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_p_spec(datatype, input_reshape_weight_p_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_p2i_spec(datatype, input_reshape_weight_p2i_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_p2o_spec(datatype, input_reshape_weight_p2o_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec input_reshape_weight_p2f_spec(datatype, input_reshape_weight_p2f_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                                                                                                
      tim::vx::TensorSpec output_spec(datatype, output_shape,
                                   tim::vx::TensorAttribute::TRANSIENT); 
      tim::vx::TensorSpec output_reshape_spec(datatype, output_reshape_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);                             
      tim::vx::TensorSpec output_hstate_spec(datatype, output_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec output_reshape_hstate_spec(datatype, output_reshape_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);                              
      tim::vx::TensorSpec output_cstate_spec(datatype, output_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);
      tim::vx::TensorSpec output_reshape_cstate_spec(datatype, output_reshape_hstate_shape,
                                    tim::vx::TensorAttribute::TRANSIENT);


      auto input_fw_weight_i_tensor = graph_->CreateTensor(input_weight_i_spec);
      auto input_fw_reshape_weight_i_tensor = graph_->CreateTensor(input_reshape_weight_i_spec);
      auto input_fw_reshape_weight_i2i_tensor = graph_->CreateTensor(input_reshape_weight_i2i_spec);
      auto input_fw_reshape_weight_i2o_tensor = graph_->CreateTensor(input_reshape_weight_i2o_spec);
      auto input_fw_reshape_weight_i2f_tensor = graph_->CreateTensor(input_reshape_weight_i2f_spec);
      auto input_fw_reshape_weight_i2c_tensor = graph_->CreateTensor(input_reshape_weight_i2c_spec);
      auto input_fw_weight_r_tensor = graph_->CreateTensor(input_weight_r_spec);
      auto input_fw_reshape_weight_r_tensor = graph_->CreateTensor(input_reshape_weight_r_spec);
      auto input_fw_reshape_weight_r2i_tensor = graph_->CreateTensor(input_reshape_weight_r2i_spec);
      auto input_fw_reshape_weight_r2o_tensor = graph_->CreateTensor(input_reshape_weight_r2o_spec);
      auto input_fw_reshape_weight_r2f_tensor = graph_->CreateTensor(input_reshape_weight_r2f_spec);
      auto input_fw_reshape_weight_r2c_tensor = graph_->CreateTensor(input_reshape_weight_r2c_spec);
      auto input_fw_bias_tensor = graph_->CreateTensor(input_bias_spec);
      auto input_fw_reshape_bias_tensor = graph_->CreateTensor(input_reshape_bias_spec);
      auto input_fw_reshape_bias_i2i_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_fw_reshape_bias_i2o_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_fw_reshape_bias_i2f_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_fw_reshape_bias_i2c_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_fw_reshape_bias_r2i_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_fw_reshape_bias_r2o_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_fw_reshape_bias_r2f_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_fw_reshape_bias_r2c_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_fw_hstate_tensor = graph_->CreateTensor(input_hstate_spec);
      auto input_fw_reshape_hstate_tensor = graph_->CreateTensor(input_reshape_hstate_spec);
      auto input_fw_cstate_tensor = graph_->CreateTensor(input_cstate_spec);
      auto input_fw_reshape_cstate_tensor = graph_->CreateTensor(input_reshape_cstate_spec);
      auto input_fw_weight_p_tensor = graph_->CreateTensor(input_weight_p_spec);
      auto input_fw_reshape_weight_p_tensor = graph_->CreateTensor(input_reshape_weight_p_spec);
      auto input_fw_reshape_weight_p2i_tensor = graph_->CreateTensor(input_reshape_weight_p2i_spec);
      auto input_fw_reshape_weight_p2o_tensor = graph_->CreateTensor(input_reshape_weight_p2o_spec);
      auto input_fw_reshape_weight_p2f_tensor = graph_->CreateTensor(input_reshape_weight_p2f_spec);
      auto output_fw_tensor = graph_->CreateTensor(output_spec);
      auto output_fw_reshape_tensor = graph_->CreateTensor(output_reshape_spec);
      auto output_fw_hstate_tensor = graph_->CreateTensor(output_hstate_spec);
      auto output_fw_reshape_hstate_tensor = graph_->CreateTensor(output_reshape_hstate_spec);
      auto output_fw_cstate_tensor = graph_->CreateTensor(output_cstate_spec);
      auto output_fw_reshape_cstate_tensor = graph_->CreateTensor(output_reshape_cstate_spec);


      auto input_bw_weight_i_tensor = graph_->CreateTensor(input_weight_i_spec);
      auto input_bw_reshape_weight_i_tensor = graph_->CreateTensor(input_reshape_weight_i_spec);
      auto input_bw_reshape_weight_i2i_tensor = graph_->CreateTensor(input_reshape_weight_i2i_spec);
      auto input_bw_reshape_weight_i2o_tensor = graph_->CreateTensor(input_reshape_weight_i2o_spec);
      auto input_bw_reshape_weight_i2f_tensor = graph_->CreateTensor(input_reshape_weight_i2f_spec);
      auto input_bw_reshape_weight_i2c_tensor = graph_->CreateTensor(input_reshape_weight_i2c_spec);
      auto input_bw_weight_r_tensor = graph_->CreateTensor(input_weight_r_spec);
      auto input_bw_reshape_weight_r_tensor = graph_->CreateTensor(input_reshape_weight_r_spec);
      auto input_bw_reshape_weight_r2i_tensor = graph_->CreateTensor(input_reshape_weight_r2i_spec);
      auto input_bw_reshape_weight_r2o_tensor = graph_->CreateTensor(input_reshape_weight_r2o_spec);
      auto input_bw_reshape_weight_r2f_tensor = graph_->CreateTensor(input_reshape_weight_r2f_spec);
      auto input_bw_reshape_weight_r2c_tensor = graph_->CreateTensor(input_reshape_weight_r2c_spec);
      auto input_bw_bias_tensor = graph_->CreateTensor(input_bias_spec);
      auto input_bw_reshape_bias_tensor = graph_->CreateTensor(input_reshape_bias_spec);
      auto input_bw_reshape_bias_i2i_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_bw_reshape_bias_i2o_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_bw_reshape_bias_i2f_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_bw_reshape_bias_i2c_tensor = graph_->CreateTensor(input_reshape_bias_i_spec);
      auto input_bw_reshape_bias_r2i_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_bw_reshape_bias_r2o_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_bw_reshape_bias_r2f_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_bw_reshape_bias_r2c_tensor = graph_->CreateTensor(input_reshape_bias_r_spec);
      auto input_bw_hstate_tensor = graph_->CreateTensor(input_hstate_spec);
      auto input_bw_reshape_hstate_tensor = graph_->CreateTensor(input_reshape_hstate_spec);
      auto input_bw_cstate_tensor = graph_->CreateTensor(input_cstate_spec);
      auto input_bw_reshape_cstate_tensor = graph_->CreateTensor(input_reshape_cstate_spec);
      auto input_bw_weight_p_tensor = graph_->CreateTensor(input_weight_p_spec);
      auto input_bw_reshape_weight_p_tensor = graph_->CreateTensor(input_reshape_weight_p_spec);
      auto input_bw_reshape_weight_p2i_tensor = graph_->CreateTensor(input_reshape_weight_p2i_spec);
      auto input_bw_reshape_weight_p2o_tensor = graph_->CreateTensor(input_reshape_weight_p2o_spec);
      auto input_bw_reshape_weight_p2f_tensor = graph_->CreateTensor(input_reshape_weight_p2f_spec);
      auto output_bw_tensor = graph_->CreateTensor(output_spec);
      auto output_bw_reshape_tensor = graph_->CreateTensor(output_reshape_spec);
      auto output_bw_hstate_tensor = graph_->CreateTensor(output_hstate_spec);
      auto output_bw_reshape_hstate_tensor = graph_->CreateTensor(output_reshape_hstate_spec);
      auto output_bw_cstate_tensor = graph_->CreateTensor(output_cstate_spec);
      auto output_bw_reshape_cstate_tensor = graph_->CreateTensor(output_reshape_cstate_spec);

      std::vector<uint32_t> slices_directions = {1, 1};
      split_weight_ = graph_->CreateOperation<tim::vx::ops::Split>(2, slices_directions);
      reshape_fw_weight_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_i_shape);
      reshape_bw_weight_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_i_shape);
      std::vector<uint32_t> slices_i = {num_units, num_units, num_units, num_units};
      split_fw_i_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_i);
      split_bw_i_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_i);

      split_recurrent_ = graph_->CreateOperation<tim::vx::ops::Split>(2, slices_directions);
      reshape_fw_recurrent_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_r_shape);
      reshape_bw_recurrent_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_r_shape);
      std::vector<uint32_t> slices_r = {num_units, num_units, num_units, num_units};
      split_fw_r_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_r);
      split_bw_r_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_r);

      split_bias_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_directions);
      reshape_fw_bias_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_bias_shape);
      reshape_bw_bias_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_bias_shape);
      std::vector<uint32_t> slices_b = {num_units, num_units, num_units, num_units, num_units, num_units, num_units, num_units};
      split_fw_b_ = graph_->CreateOperation<tim::vx::ops::Split>(0, slices_b);
      split_bw_b_ = graph_->CreateOperation<tim::vx::ops::Split>(0, slices_b);

      lstm_ = graph_->CreateOperation<tim::vx::ops::BidirectionalSequenceLstm>(cell_clip_,  0.0,  act_type_,  0.0, time_major_, recurrent_act_type_, true);
      
      reshape_fw_out_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_shape);
      reshape_fw_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_hstate_shape);
      reshape_fw_out_cstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_cstate_shape);

      reshape_bw_out_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_shape);
      reshape_bw_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_hstate_shape);
      reshape_bw_out_cstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(output_reshape_cstate_shape);

      if(time_major_)
      {
          concat_output_ = graph_->CreateOperation<tim::vx::ops::Concat>(2, 2);
      }
      else
      {
          concat_output_ = graph_->CreateOperation<tim::vx::ops::Concat>(1, 2);
      }
      
      concat_out_hstate_ = graph_->CreateOperation<tim::vx::ops::Concat>(2, 2);
      concat_out_cstate_ = graph_->CreateOperation<tim::vx::ops::Concat>(2, 2);

      split_weight_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_I]}).
      BindOutputs({input_fw_weight_i_tensor, input_bw_weight_i_tensor});
      reshape_fw_weight_->BindInputs({input_fw_weight_i_tensor}).
      BindOutputs({input_fw_reshape_weight_i_tensor});
      reshape_bw_weight_->BindInputs({input_bw_weight_i_tensor}).
      BindOutputs({input_bw_reshape_weight_i_tensor});
      split_fw_i_->BindInput({input_fw_reshape_weight_i_tensor}).
      BindOutputs({input_fw_reshape_weight_i2i_tensor,
                  input_fw_reshape_weight_i2o_tensor,
                  input_fw_reshape_weight_i2f_tensor,
                  input_fw_reshape_weight_i2c_tensor});
      split_bw_i_->BindInput({input_bw_reshape_weight_i_tensor}).
      BindOutputs({input_bw_reshape_weight_i2i_tensor,
                  input_bw_reshape_weight_i2o_tensor,
                  input_bw_reshape_weight_i2f_tensor,
                  input_bw_reshape_weight_i2c_tensor});


      split_recurrent_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_R]}).
      BindOutputs({input_fw_weight_r_tensor, input_bw_weight_r_tensor});
      reshape_fw_recurrent_->BindInputs({input_fw_weight_r_tensor}).
      BindOutputs({input_fw_reshape_weight_r_tensor});
      reshape_bw_recurrent_->BindInputs({input_bw_weight_r_tensor}).
      BindOutputs({input_bw_reshape_weight_r_tensor});
      split_fw_r_->BindInput({input_fw_reshape_weight_r_tensor}).
      BindOutputs({input_fw_reshape_weight_r2i_tensor,
                  input_fw_reshape_weight_r2o_tensor,
                  input_fw_reshape_weight_r2f_tensor,
                  input_fw_reshape_weight_r2c_tensor});
      split_bw_r_->BindInput({input_bw_reshape_weight_r_tensor}).
      BindOutputs({input_bw_reshape_weight_r2i_tensor,
                  input_bw_reshape_weight_r2o_tensor,
                  input_bw_reshape_weight_r2f_tensor,
                  input_bw_reshape_weight_r2c_tensor});


      split_bias_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_BIAS]}). 
      BindOutputs({input_fw_bias_tensor, input_bw_bias_tensor});
      reshape_fw_bias_->BindInputs({input_fw_bias_tensor}). 
      BindOutputs({input_fw_reshape_bias_tensor});
      reshape_bw_bias_->BindInputs({input_bw_bias_tensor}). 
      BindOutputs({input_bw_reshape_bias_tensor});
      split_fw_b_->BindInputs({input_fw_reshape_bias_tensor}). 
      BindOutputs({input_fw_reshape_bias_i2i_tensor,
                  input_fw_reshape_bias_i2o_tensor,
                  input_fw_reshape_bias_i2f_tensor,
                  input_fw_reshape_bias_i2c_tensor,
                  input_fw_reshape_bias_r2i_tensor,
                  input_fw_reshape_bias_r2o_tensor,
                  input_fw_reshape_bias_r2f_tensor,
                  input_fw_reshape_bias_r2c_tensor});
      split_bw_b_->BindInputs({input_bw_reshape_bias_tensor}). 
      BindOutputs({input_bw_reshape_bias_i2i_tensor,
                  input_bw_reshape_bias_i2o_tensor,
                  input_bw_reshape_bias_i2f_tensor,
                  input_bw_reshape_bias_i2c_tensor,
                  input_bw_reshape_bias_r2i_tensor,
                  input_bw_reshape_bias_r2o_tensor,
                  input_bw_reshape_bias_r2f_tensor,
                  input_bw_reshape_bias_r2c_tensor});

      if(in_tensors_[BI_LSTM_EXT_INPUT_H_STATE]->IsPlaceHolder())
      {
          input_fw_reshape_hstate_tensor = graph_->CreateTensorPlaceHolder();
          input_bw_reshape_hstate_tensor = graph_->CreateTensorPlaceHolder();
      }
      else
      {   
          split_hstate_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_H_STATE]}). 
          BindOutputs({input_fw_hstate_tensor, input_bw_hstate_tensor});
          reshape_fw_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_hstate_shape);
          reshape_fw_hstate_->BindInputs({input_fw_hstate_tensor}). 
          BindOutputs({input_fw_reshape_hstate_tensor});
          reshape_bw_hstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_hstate_shape);
          reshape_bw_hstate_->BindInputs({input_bw_hstate_tensor}). 
          BindOutputs({input_bw_reshape_hstate_tensor});
      }
      
      if(in_tensors_[BI_LSTM_EXT_INPUT_C_STATE]->IsPlaceHolder())
      {
          input_fw_reshape_cstate_tensor = graph_->CreateTensorPlaceHolder();
          input_bw_reshape_cstate_tensor = graph_->CreateTensorPlaceHolder();
      }
      else
      {   
          split_cstate_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_C_STATE]}). 
          BindOutputs({input_fw_cstate_tensor, input_bw_cstate_tensor});
          reshape_fw_cstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_cstate_shape);
          reshape_fw_cstate_->BindInputs({input_fw_cstate_tensor}). 
          BindOutputs({input_fw_reshape_cstate_tensor});
          reshape_bw_cstate_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_cstate_shape);
          reshape_bw_cstate_->BindInputs({input_bw_cstate_tensor}). 
          BindOutputs({input_bw_reshape_cstate_tensor});
      }
      
      if(in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_P]->IsPlaceHolder())
      {
          input_fw_reshape_weight_p2i_tensor = graph_->CreateTensorPlaceHolder();
          input_fw_reshape_weight_p2o_tensor = graph_->CreateTensorPlaceHolder();
          input_fw_reshape_weight_p2f_tensor = graph_->CreateTensorPlaceHolder();
          input_bw_reshape_weight_p2i_tensor = graph_->CreateTensorPlaceHolder();
          input_bw_reshape_weight_p2o_tensor = graph_->CreateTensorPlaceHolder();
          input_bw_reshape_weight_p2f_tensor = graph_->CreateTensorPlaceHolder();
      }
      else
      {
          split_p_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_WEIGHT_P]}). 
          BindOutputs({input_fw_weight_p_tensor, input_bw_weight_p_tensor});
          reshape_fw_p_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_p_shape);
          reshape_fw_p_->BindInput({input_fw_weight_p_tensor}). 
          BindOutputs({input_fw_reshape_weight_p_tensor});
          std::vector<uint32_t> slices_p = {num_units, num_units, num_units};
          split_fw_p_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_p);
          split_fw_p_->BindInputs({input_fw_reshape_weight_p_tensor}). 
          BindOutputs({input_fw_reshape_weight_p2i_tensor,
                       input_fw_reshape_weight_p2o_tensor,
                       input_fw_reshape_weight_p2f_tensor});
          
          reshape_bw_p_ = graph_->CreateOperation<tim::vx::ops::Reshape>(input_reshape_weight_p_shape);
          reshape_bw_p_->BindInput({input_bw_weight_p_tensor}). 
          BindOutputs({input_bw_reshape_weight_p_tensor});
          split_bw_p_ = graph_->CreateOperation<tim::vx::ops::Split>(1, slices_p);
          split_bw_p_->BindInputs({input_bw_reshape_weight_p_tensor}). 
          BindOutputs({input_bw_reshape_weight_p2i_tensor,
                       input_bw_reshape_weight_p2o_tensor,
                       input_bw_reshape_weight_p2f_tensor});
      }

      lstm_->BindInputs({in_tensors_[BI_LSTM_EXT_INPUT_INPUT], 
                        input_fw_reshape_weight_i2i_tensor,
                        input_fw_reshape_weight_i2f_tensor,
                        input_fw_reshape_weight_i2c_tensor,
                        input_fw_reshape_weight_i2o_tensor,

                        input_fw_reshape_weight_r2i_tensor,
                        input_fw_reshape_weight_r2f_tensor,
                        input_fw_reshape_weight_r2c_tensor,
                        input_fw_reshape_weight_r2o_tensor,

                        input_fw_reshape_weight_p2i_tensor,
                        input_fw_reshape_weight_p2f_tensor,
                        input_fw_reshape_weight_p2o_tensor,

                        input_fw_reshape_bias_i2i_tensor,
                        input_fw_reshape_bias_i2f_tensor,
                        input_fw_reshape_bias_i2c_tensor,
                        input_fw_reshape_bias_i2o_tensor,

                        input_fw_reshape_bias_r2i_tensor,
                        input_fw_reshape_bias_r2f_tensor,
                        input_fw_reshape_bias_r2c_tensor,
                        input_fw_reshape_bias_r2o_tensor,

                        // optional for projection
                        graph_->CreateTensorPlaceHolder(),       /*fw_weight_prj*/
                        graph_->CreateTensorPlaceHolder(),       /*fw_bias_prj*/

                        input_bw_reshape_weight_i2i_tensor,
                        input_bw_reshape_weight_i2f_tensor,
                        input_bw_reshape_weight_i2c_tensor,
                        input_bw_reshape_weight_i2o_tensor,

                        input_bw_reshape_weight_r2i_tensor,
                        input_bw_reshape_weight_r2f_tensor,
                        input_bw_reshape_weight_r2c_tensor,
                        input_bw_reshape_weight_r2o_tensor,

                        input_bw_reshape_weight_p2i_tensor,
                        input_bw_reshape_weight_p2f_tensor,
                        input_bw_reshape_weight_p2o_tensor,

                        input_bw_reshape_bias_i2i_tensor,
                        input_bw_reshape_bias_i2f_tensor,
                        input_bw_reshape_bias_i2c_tensor,
                        input_bw_reshape_bias_i2o_tensor,

                        input_bw_reshape_bias_r2i_tensor,
                        input_bw_reshape_bias_r2f_tensor,
                        input_bw_reshape_bias_r2c_tensor,
                        input_bw_reshape_bias_r2o_tensor,

                        // optional for projection
                        graph_->CreateTensorPlaceHolder(),       /*bw_weight_prj*/
                        graph_->CreateTensorPlaceHolder(),       /*bw_bias_prj*/

                        input_fw_reshape_hstate_tensor,
                        input_fw_reshape_cstate_tensor,

                        input_bw_reshape_hstate_tensor,
                        input_bw_reshape_cstate_tensor,

                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),       // AUX

                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),
                        graph_->CreateTensorPlaceHolder(),       // Layer_norm

                        });
      lstm_->BindOutputs({output_fw_tensor,
                          output_fw_hstate_tensor,
                          output_fw_cstate_tensor,
                          output_bw_tensor,
                          output_bw_hstate_tensor,
                          output_bw_cstate_tensor});

      reshape_fw_out_->BindInputs({output_fw_tensor}). 
      BindOutputs({output_fw_reshape_tensor});
      reshape_fw_out_hstate_->BindInputs({output_fw_hstate_tensor}). 
      BindOutputs({output_fw_reshape_hstate_tensor});
      reshape_fw_out_cstate_->BindInputs({output_fw_cstate_tensor}). 
      BindOutputs({output_fw_reshape_cstate_tensor});
      reshape_bw_out_->BindInputs({output_bw_tensor}). 
      BindOutputs({output_bw_reshape_tensor});
      reshape_bw_out_hstate_->BindInputs({output_bw_hstate_tensor}). 
      BindOutputs({output_bw_reshape_hstate_tensor});
      reshape_bw_out_cstate_->BindInputs({output_bw_cstate_tensor}). 
      BindOutputs({output_bw_reshape_cstate_tensor});

      concat_out_hstate_->BindInputs({output_fw_reshape_hstate_tensor, output_bw_reshape_hstate_tensor});
      concat_out_cstate_->BindInputs({output_fw_reshape_cstate_tensor, output_bw_reshape_cstate_tensor});
      concat_output_->BindInputs({output_fw_reshape_tensor, output_bw_reshape_tensor});

    }
    this->input_tensor_index++;
    return *this;
  }

  BidirectionalSequenceLstmExtImpl& BindOutput(const std::shared_ptr<Tensor>& tensor) override {
    out_tensors_[output_tensor_index] = tensor;

    if (this->output_tensor_index == BI_LSTM_EXT_OUT_CNT - 1) {
      concat_output_->BindOutput(out_tensors_[BI_LSTM_EXT_OUTPUT_OUTPUT]);
      concat_out_hstate_->BindOutput(out_tensors_[BI_LSTM_EXT_OUTPUT_H_STATE]);
      concat_out_cstate_->BindOutput(out_tensors_[BI_LSTM_EXT_OUTPUT_C_STATE]);
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
  float cell_clip_;
  tim::vx::ops::BidirectionalSequenceLstm::ActivationType act_type_;
  bool time_major_;
  tim::vx::ops::BidirectionalSequenceLstm::ActivationType recurrent_act_type_;
  
  std::shared_ptr<tim::vx::Operation> split_weight_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_weight_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_weight_;
  std::shared_ptr<tim::vx::Operation> split_fw_i_;
  std::shared_ptr<tim::vx::Operation> split_bw_i_;

  std::shared_ptr<tim::vx::Operation> split_recurrent_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_recurrent_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_recurrent_;
  std::shared_ptr<tim::vx::Operation> split_fw_r_;
  std::shared_ptr<tim::vx::Operation> split_bw_r_;
  
  std::shared_ptr<tim::vx::Operation> split_bias_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_bias_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_bias_;
  std::shared_ptr<tim::vx::Operation> split_fw_b_;
  std::shared_ptr<tim::vx::Operation> split_bw_b_;

  std::shared_ptr<tim::vx::Operation> split_p_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_p_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_p_;
  std::shared_ptr<tim::vx::Operation> split_fw_p_;
  std::shared_ptr<tim::vx::Operation> split_bw_p_;
  
  std::shared_ptr<tim::vx::Operation> split_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_hstate_;


  std::shared_ptr<tim::vx::Operation> split_cstate_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_cstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_cstate_;
  
  std::shared_ptr<tim::vx::Operation> reshape_fw_out_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_out_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_fw_out_cstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_out_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_out_hstate_;
  std::shared_ptr<tim::vx::Operation> reshape_bw_out_cstate_;

  std::shared_ptr<tim::vx::Operation> concat_output_;
  std::shared_ptr<tim::vx::Operation> concat_out_hstate_;
  std::shared_ptr<tim::vx::Operation> concat_out_cstate_;
  std::shared_ptr<tim::vx::Operation> lstm_;

  std::array<std::shared_ptr<tim::vx::Tensor>, BI_LSTM_EXT_INPUT_CNT> in_tensors_;
  std::array<std::shared_ptr<tim::vx::Tensor>, BI_LSTM_EXT_OUT_CNT> out_tensors_;
};

BidirectionalSequenceLstmExt::BidirectionalSequenceLstmExt(Graph* graph, 
                                                          float cell_clip,
                                                          tim::vx::ops::BidirectionalSequenceLstm::ActivationType act_type,
                                                          bool time_major,
                                                          tim::vx::ops::BidirectionalSequenceLstm::ActivationType recurrent_act_type)
                                                          : cell_clip_(cell_clip),
                                                          act_type_(act_type),
                                                          time_major_(time_major),
                                                          recurrent_act_type_(recurrent_act_type){
  impl_ = std::make_unique<BidirectionalSequenceLstmExtImpl>(graph, cell_clip, act_type, time_major, recurrent_act_type, DataLayout::ANY);
}

std::shared_ptr<Operation> BidirectionalSequenceLstmExt::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<BidirectionalSequenceLstmExt>(this->cell_clip_, this->act_type_, this->time_major_, this->recurrent_act_type_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim
