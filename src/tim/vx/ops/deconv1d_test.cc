/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/deconv1d.h"

#include "gtest/gtest.h"

TEST(DeConv1d, no_bias_layout_whcn_depthwise_shape_3_2_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape ({3, 2, 1});  //whcn
    tim::vx::ShapeType kernel_shape({3, 2, 1});  //whc1 same as depthwise convolution
    tim::vx::ShapeType output_shape({5, 2, 1});  //whcn

    tim::vx::TensorSpec input_spec  (tim::vx::DataType::FLOAT32, input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec kernel_spec (tim::vx::DataType::FLOAT32, kernel_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec (tim::vx::DataType::FLOAT32, output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto kernel_tensor = graph->CreateTensor(kernel_spec);

    std::vector<float> input_data = { 3.0f, 9.0f, 3.0f,
                                      7.0f, 5.0f, 9.0f, };
    std::vector<float> kernel_data = { 9.0f, 0.0f, 1.0f,
                                       3.0f, 0.0f, 0.0f, };

    std::vector<float> golden = {
                        27.0f, 81.0f, 30.0f, 9.0f, 3.0f,
                        21.0f, 15.0f, 27.0f, 0.0f, 0.0f, };

    std::vector<float> output_data(golden.size());

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()*4));
    EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(), kernel_data.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
        2, tim::vx::PadType::SAME, 3, 1, 1, std::array<uint32_t, 2>({0, 0}), 2);
    (*op).BindInputs({input_tensor, kernel_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));

    EXPECT_EQ(golden, output_data) << "Result mismatch";
}

TEST(DeConv1d, layout_whcn_shape_3_1_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape ({3, 1, 1});
    tim::vx::ShapeType kernel_shape({3, 1, 1});
    tim::vx::ShapeType output_shape({5, 1, 1});
    tim::vx::ShapeType bias_shape({1});

    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 2);
    tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
    tim::vx::Quantization bias_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);

    tim::vx::TensorSpec input_spec  (
        tim::vx::DataType::UINT8, input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec kernel_spec (
        tim::vx::DataType::UINT8, kernel_shape, tim::vx::TensorAttribute::CONSTANT, weight_quant);
    tim::vx::TensorSpec bias_spec (
        tim::vx::DataType::INT32, bias_shape, tim::vx::TensorAttribute::CONSTANT, bias_quant);
    tim::vx::TensorSpec output_spec (
        tim::vx::DataType::UINT8, output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto kernel_tensor = graph->CreateTensor(kernel_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);

    std::vector<uint8_t> input_data = {
        3, 9, 3,
    };

    std::vector<uint8_t> kernel_data = {
        9, 0, 1,
    };

    std::vector<int32_t> bias_data = {
        -5,
    };

    std::vector<uint8_t> golden = {
        24, 78, 27, 6, 0,
    };

    std::vector<uint8_t> output_data(golden.size());

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()));
    EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(), kernel_data.size()));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias_data.data(), bias_data.size() * sizeof(int32_t)));

    auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
        1, tim::vx::PadType::SAME, 3, 1, 1,
        std::array<uint32_t, 2>({0, 0,}),
        1);
    (*op).BindInputs({input_tensor, kernel_tensor, bias_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));
    EXPECT_EQ(golden, output_data) << "Result mismatch";
}