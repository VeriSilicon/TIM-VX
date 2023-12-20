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
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(RNNCell, shape_3_2_4_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    float tolerance = ctx->hasSP() ? 1e-4f : 1e-5f;

    uint32_t input_size = 3, batch_size = 2, num_units = 4;

    tim::vx::ShapeType input_shape({input_size, batch_size});
    tim::vx::ShapeType weights_shape({input_size, num_units});
    tim::vx::ShapeType recurrent_weights_shape({num_units, num_units});
    tim::vx::ShapeType bias_shape({num_units});
    tim::vx::ShapeType state_in_shape({num_units, batch_size});
    tim::vx::ShapeType output_shape({num_units, batch_size});
    tim::vx::ShapeType state_out_shape({num_units, batch_size});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);

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
    tim::vx::TensorSpec state_out_spec(tim::vx::DataType::UINT8,
        state_out_shape, tim::vx::TensorAttribute::OUTPUT,quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weights_tensor = graph->CreateTensor(weights_spec);
    auto recurrent_weights_tensor = graph->CreateTensor(recurrent_weights_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto state_out_tensor = graph->CreateTensor(state_out_spec);

    std::vector<float> in_data = {
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406, 0.73463392,
        };
    std::vector<float> weights_data = {
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406,  0.73463392,
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406,  0.73463392,
        };
    std::vector<float> recurrent_weights_data = {
        -0.31930989, 0.37613347,  0.27901134,  0.36137494,
        -1.36916667, 0.38031587,  0.21580373, 0.27072677,
        1.01580888, 0.14943552, 1.15465137,  0.09784451,
        -1.02702999, 1.39296314,  0.15785322,  0.21931258,
    };
    std::vector<float> bias_data = {
        0.01580888, 0.14943552, 0.15465137,  0.09784451,
    };
    std::vector<float> state_in_data = {
        0,0,0,0,0,0,0,0
    };
    std::vector<float> output_golden = {
        0.781534, 0.771447, 0.830002, 0.749713, 0.711524, 0.74155, 0.77355, 0.717427
    };
    std::vector<uint8_t> state_out_golden = {
        217, 214,  231, 208, 198,  206,  215, 199,
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

    auto op = graph->CreateOperation<tim::vx::ops::RNNCell>(tim::vx::ops::RNNCell::ActivationType::kSIGMOID);
    (*op).BindInputs({input_tensor, weights_tensor, bias_tensor, state_in_tensor, recurrent_weights_tensor})
         .BindOutputs({output_tensor, state_out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(output_golden.size());
    std::vector<uint8_t> state_out(state_out_golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(state_out_tensor->CopyDataFromTensor(state_out.data()));

    EXPECT_TRUE(ArraysMatch(output_golden, output, tolerance));
    EXPECT_EQ(state_out_golden, state_out);
}

TEST(RNNCell, seperate) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    float tolerance = ctx->hasSP() ? 1e-4f : 1e-5f;
    std::vector<float> in_data = {
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406, 0.73463392,
        };
    std::vector<float> weights_data = {
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406,  0.73463392,
        0.12609188,  0.46347019, 0.89598465,
        0.35867718,  0.36897406,  0.73463392,
        };
    std::vector<float> recurrent_weights_data = {
        -0.31930989, 0.37613347,  0.27901134,  0.36137494,
        -1.36916667, 0.38031587,  0.21580373, 0.27072677,
        1.01580888, 0.14943552, 1.15465137,  0.09784451,
        -1.02702999, 1.39296314,  0.15785322,  0.21931258,
    };
    std::vector<float> bias_data = {
        0.01580888, 0.14943552, 0.15465137,  0.09784451,
    };
    std::vector<float> state_in_data = {
        0,0,0,0,0,0,0,0
    };
    std::vector<float> output_golden = {
        0.781534, 0.771447, 0.830002, 0.749713, 0.711524, 0.74155, 0.77355, 0.717427
    };
    std::vector<uint8_t> state_out_golden = {
        217, 214,  231, 208, 198,  206, 215, 199,
    };
    uint32_t input_size = 3, batch_size = 2, num_units = 4;

    tim::vx::ShapeType input_shape({input_size, batch_size});
    tim::vx::ShapeType weights_shape({input_size, num_units});
    tim::vx::ShapeType recurrent_weights_shape({num_units, num_units});
    tim::vx::ShapeType bias_shape({num_units});
    tim::vx::ShapeType state_in_shape({num_units, batch_size});

    tim::vx::ShapeType FC1_shape({num_units, batch_size});
    tim::vx::ShapeType add_shape({num_units, batch_size});
    tim::vx::ShapeType FC2_shape({num_units, batch_size});
    tim::vx::ShapeType activation_shape({num_units, batch_size});
    tim::vx::ShapeType convert_shape({num_units, batch_size});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);

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

    tim::vx::TensorSpec FC1_spec(tim::vx::DataType::FLOAT32,
        FC1_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec FC2_spec(tim::vx::DataType::FLOAT32,
        FC2_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec add_spec(tim::vx::DataType::FLOAT32,
        add_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec activation_out_spec(tim::vx::DataType::FLOAT32,
        activation_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec convert_spec(tim::vx::DataType::UINT8,
        convert_shape, tim::vx::TensorAttribute::OUTPUT, quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weights_tensor = graph->CreateTensor(weights_spec);
    auto recurrent_weights_tensor = graph->CreateTensor(recurrent_weights_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);

    auto FC1_tensor = graph->CreateTensor(FC1_spec);
    auto FC2_tensor = graph->CreateTensor(FC2_spec);
    auto add_tensor = graph->CreateTensor(add_spec);
    auto activation_out_tensor = graph->CreateTensor(activation_out_spec);
    auto convert_tensor = graph->CreateTensor(convert_spec);

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

    auto op1 = graph->CreateOperation<tim::vx::ops::FullyConnected>(0,4);
    (*op1).BindInputs({input_tensor, weights_tensor, bias_tensor})
         .BindOutputs({FC1_tensor});
    auto op2 = graph->CreateOperation<tim::vx::ops::FullyConnected>(0,4);
    (*op2).BindInputs({state_in_tensor, recurrent_weights_tensor})
         .BindOutputs({FC2_tensor});
    auto op3 = graph->CreateOperation<tim::vx::ops::Add>();
    (*op3).BindInputs({FC1_tensor, FC2_tensor})
         .BindOutputs({add_tensor});
    auto op4 = graph->CreateOperation<tim::vx::ops::Tanh>();
    (*op4).BindInputs({add_tensor})
         .BindOutputs({activation_out_tensor});
    auto op5 = graph->CreateOperation<tim::vx::ops::DataConvert>();
    (*op5).BindInputs({activation_out_tensor})
         .BindOutputs({convert_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(output_golden.size());
    std::vector<uint8_t> state_out(state_out_golden.size());
    EXPECT_TRUE(activation_out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(convert_tensor->CopyDataFromTensor(state_out.data()));
    EXPECT_TRUE(ArraysMatch(output_golden, output, tolerance));
    EXPECT_EQ(state_out_golden, state_out);
}