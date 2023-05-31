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
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/vx/types.h"
#include "gtest/gtest.h"
#include "test_utils.h"


TEST(BidirectionalSequenceRnnExt, shape_2_3_2_float_sigmoid) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t input_size = 2, batch_size = 3, num_units = 4;

    tim::vx::ShapeType input_shape({input_size, batch_size, 2});
    tim::vx::ShapeType weights_shape({input_size, num_units, 2});
    tim::vx::ShapeType recurrent_weights_shape({num_units, num_units, 2});
    tim::vx::ShapeType bias_shape({num_units*2, 2});
    tim::vx::ShapeType state_in_shape({num_units, batch_size, 2});
    tim::vx::ShapeType output_shape({num_units, batch_size, 2, 2});
    tim::vx::ShapeType state_out_shape({num_units, batch_size, 2});

    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_spec(tim::vx::DataType::FLOAT32,
        weights_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec recurrent_weights_spec(tim::vx::DataType::FLOAT32,
        recurrent_weights_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32,
        bias_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32,
        state_in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec state_out_spec(tim::vx::DataType::FLOAT32,
        state_out_shape, tim::vx::TensorAttribute::OUTPUT);


    auto input_tensor = graph->CreateTensor(input_spec);
    auto weights_tensor = graph->CreateTensor(weights_spec);
    auto recurrent_weights_tensor = graph->CreateTensor(recurrent_weights_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto state_out_tensor = graph->CreateTensor(state_out_spec);
    
   std::vector<float> in_data = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
        };
    std::vector<float> weights_data = {
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1,
        0.1, 0.1
        };
    std::vector<float> recurrent_weights_data = {
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1,
    };
    std::vector<float> bias_data = {
        0.1, 0.1, 0.1, 0.1, 
        0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.1, 0.1,
        0.0, 0.0, 0.0, 0.0,
    };
    std::vector<float> state_in_data = {
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0
    };
    std::vector<float> output_golden = {
        0.5986, 0.5986, 0.5986, 0.5986,
        0.6899, 0.6899, 0.6899, 0.6899,
        0.7685, 0.7685, 0.7685, 0.7685,
        0.6754, 0.6754, 0.6754, 0.6754,
        0.7599, 0.7599, 0.7599, 0.7599,
        0.8273, 0.8273, 0.8273, 0.8273, 
        0.8628, 0.8628, 0.8628, 0.8628,
        0.9068, 0.9068, 0.9068, 0.9068,
        0.9374, 0.9374, 0.9374, 0.9374,
        0.8320, 0.8320, 0.8320, 0.8320,
        0.8807, 0.8807, 0.8807, 0.8807,
        0.9168, 0.9168, 0.9168, 0.9168,
    };
    std::vector<float> state_out_golden = {
        0.8628, 0.8628, 0.8628, 0.8628,
        0.9068, 0.9068, 0.9068, 0.9068,
        0.9374, 0.9374, 0.9374, 0.9374,
        0.6754, 0.6754, 0.6754, 0.6754,
        0.7599, 0.7599, 0.7599, 0.7599,
        0.8273, 0.8273, 0.8273, 0.8273
    };


    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weights_tensor->CopyDataToTensor(
        weights_data.data(), weights_data.size() * sizeof(float)));
    EXPECT_TRUE(recurrent_weights_tensor->CopyDataToTensor(
        recurrent_weights_data.data(), recurrent_weights_data.size() * sizeof(float)));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(
        bias_data.data(), bias_data.size() * sizeof(float)));
    EXPECT_TRUE(state_in_tensor->CopyDataToTensor(
        state_in_data.data(), state_in_data.size() * sizeof(float)));
   
    auto op = graph->CreateOperation<tim::vx::ops::BidirectionalSequenceRnnExt>(tim::vx::ops::BidirectionalSequenceRnn::ActivationType::kSIGMOID);
    (*op).BindInputs({input_tensor, weights_tensor,  recurrent_weights_tensor, bias_tensor, state_in_tensor})
         .BindOutputs({state_out_tensor, output_tensor});
    graph->PrintGraph();
    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(output_golden.size());
    std::vector<float> state_out(state_out_golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(state_out_tensor->CopyDataFromTensor(state_out.data()));


    EXPECT_TRUE(ArraysMatch(output_golden, output,1e-3f));
    EXPECT_TRUE(ArraysMatch(state_out_golden, state_out,1e-3f));
}