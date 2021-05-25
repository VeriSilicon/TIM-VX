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
#include "tim/vx/ops/conv1d.h"

#include "gtest/gtest.h"

namespace {
template<typename T>
::testing::AssertionResult ArraysMatch(const std::vector<T>& expected,
                                       const std::vector<T>& actual,
                                       T abs_error){
    for (size_t i = 0; i < expected.size(); ++i){
        EXPECT_NEAR(expected[i], actual[i], abs_error) << "at index:" << i;
    }

    return ::testing::AssertionSuccess();
}
}

TEST(Conv1d, shape_3_6_1_float_ksize_1_stride_1_weights_3_no_bias_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 6, 1});
    tim::vx::ShapeType param_shape({1,6,3});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -1,    0,   1,
        -1.5,  0.5, 1.5,
        -2,   -0.5, 2,
        -2.5,  0,   2.5,
        -3,    0.5, 3,
        -3.5,  0.5, 3.5,
        };
    std::vector<float> weight = {
        -3,   -2, -1.5, 1.5, 2, 3,
        -2.5, -2, -1.5, 1.5, 2, 2.5,
        -2.5, -2, 0,    0,   2, 2.5,
    };
    std::vector<float> golden = {
        -11.25, 2.25, 11.25,
        -10,    2,    10,
        -9.25,  1.25, 9.25,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Conv1d>(3, tim::vx::PadType::VALID,
        1, 1, 1);
    (*op).BindInputs({input_tensor, weight_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Conv1d, shape_6_2_1_uint8_ksize_6_stride_1_weights_2_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({6, 2, 1});
    tim::vx::ShapeType output_shape({1, 2, 1});
    tim::vx::ShapeType param_shape({6, 2, 2});
    tim::vx::ShapeType bias_shape({2});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 6);
    tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 22);
    tim::vx::Quantization bias_quant(tim::vx::QuantType::ASYMMETRIC, 0.0625, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8,
                            param_shape, tim::vx::TensorAttribute::CONSTANT, weight_quant);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32,
                            bias_shape, tim::vx::TensorAttribute::CONSTANT, bias_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        4,  5,  6,  6,  7,  8,
        0,  2,  4,  8, 10, 12,
        };
    std::vector<uint8_t> weight = {
        12, 14,
        16, 28,
        30, 32,
         8, 10,
        12, 32,
        34, 36,
         4,  6,
         8, 36,
        38, 40,
         0,  2,
         4, 40,
        42, 44,
    };
    std::vector<int32_t> bias = {
        -20, 100,
    };
    std::vector<uint8_t> golden = {
        85, 175,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size()));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias.data(), bias.size() * sizeof(int32_t)));

    auto op = graph->CreateOperation<tim::vx::ops::Conv1d>(2, tim::vx::PadType::VALID, 6, 1, 1);
    (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, static_cast<uint8_t>(0)));
}

TEST(Conv1d, shape_6_2_1_uint8_ksize_3_stride_1_pad_1_weights_2_no_bias_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({6, 2, 1});
    tim::vx::ShapeType output_shape({3, 2, 1});
    tim::vx::ShapeType param_shape({3, 2, 2});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 6);
    tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 22);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 69);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8,
                            param_shape, tim::vx::TensorAttribute::CONSTANT, weight_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        4,  4,  6,  6,  8,  8,
        0,  2,  4,  8, 10, 12,
        };
    std::vector<uint8_t> weight = {
        12, 14, 16,
         8, 10, 12,
         4,  6,  8,
         0,  2,  4,
    };
    std::vector<uint8_t> golden = {
        116, 57, 28,
        148, 45,  0,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size()));

    std::array<uint32_t, 2> pad = {0, 1};
    auto op = graph->CreateOperation<tim::vx::ops::Conv1d>(
        2, tim::vx::PadType::AUTO, 3, 2, 1, pad);
    (*op).BindInputs({input_tensor, weight_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, static_cast<uint8_t>(0)));
}

#if 0
// Fail case
// Internal impl conv1d don't support multiplier, need wait for the fix.
TEST(Conv1d, shape_7_2_1_uint8_ksize_3_stride_2_multiplier_1_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({7, 2, 1});
    tim::vx::ShapeType output_shape({3, 2, 1});
    tim::vx::ShapeType param_shape({3, 1, 2});
    tim::vx::ShapeType bias_shape({2});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 6);
    tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 22);
    tim::vx::Quantization bias_quant(tim::vx::QuantType::ASYMMETRIC, 0.0625, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 39);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8,
                            param_shape, tim::vx::TensorAttribute::CONSTANT, weight_quant);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::UINT8,
                            bias_shape, tim::vx::TensorAttribute::CONSTANT, bias_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        4,  4,  6, 10,  6,  8,  8,
        0,  2,  4, 10,  8, 10, 12,
        };
    std::vector<uint8_t> weight = {
        12, 14, 16,
        8, 10, 12,
    };
    std::vector<int32_t> bias = {
        -20, 100,
    };
    std::vector<uint8_t> golden = {
        43, 26, 27,
        72, 24,  0,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size()));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias.data(), bias.size() * sizeof(int32_t)));

    auto op = graph->CreateOperation<tim::vx::ops::Conv1d>(
        2, tim::vx::PadType::AUTO, 3, 2, 1, 1);
    (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, static_cast<uint8_t>(0)));
}
#endif
