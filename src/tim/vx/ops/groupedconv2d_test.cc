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
#include "tim/vx/ops/groupedconv2d.h"

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

TEST(GroupedConv2d, shape_3_3_6_1_float_group_1_no_bias_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3,3,6,1});
    tim::vx::ShapeType param_shape({3,3,6,1});
    tim::vx::ShapeType output_shape({1,1,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
         -0.50, -0.50, -0.50,
          0.00,  1.00,  0.00,
          0.50,  0.50,  0.50,
         -1.50, -1.00, -1.00,
         -0.50,  1.00,  0.50,
          1.00,  1.00,  1.50,
         -2.50, -2.00, -2.00,
         -1.50,  1.50,  1.50,
          2.00,  2.00,  2.50,
         -3.50, -3.00, -3.00,
         -2.50,  2.50,  2.50,
          3.00,  3.00,  3.50,
         -4.50, -4.00, -4.00,
         -3.50,  3.50,  3.50,
          4.00,  4.00,  4.50,
         -5.50, -5.00, -5.00,
         -4.50,  4.50,  4.50,
          5.00,  5.00,  5.50,
        };
    std::vector<float> weight = {
        -0.50,  0.00, -0.50,
        -0.50,  0.00, -0.50,
        -0.50,  0.00, -0.50,
         1.50,  1.00, -1.50,
         1.50,  1.00, -1.50,
         1.50,  1.00, -1.50,
        -2.50, -2.00, -2.50,
        -2.50, -2.00, -2.50,
        -2.50, -2.00, -2.50,
         3.50,  3.00,  3.50,
         3.50,  3.00,  3.50,
         3.50,  3.00,  3.50,
        -4.50, -4.00, -4.50,
        -4.50, -4.00, -4.50,
        -4.50, -4.00, -4.50,
        -5.50, -5.00,  5.50,
        -5.50, -5.00,  5.50,
        -5.50, -5.00,  5.50,
    };
    std::vector<float> golden = {
        21.0
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size() * sizeof(float)));

    std::array<uint32_t, 2> dilations = {1,1};
    std::array<uint32_t, 2> strides = {1,1};
    auto op = graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
        tim::vx::PadType::VALID, strides, dilations, 1);
    (*op).BindInputs({input_tensor, weight_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(GroupedConv2d, shape_3_3_6_1_float_group_2_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3,3,6,1});
    tim::vx::ShapeType weight_shape({3,3,3,2});
    tim::vx::ShapeType bias_shape({2});
    tim::vx::ShapeType output_shape({1,1,2,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32,
                            weight_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32,
                            bias_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
         -0.50, -0.50, -0.50,
          0.00,  1.00,  0.00,
          0.50,  0.50,  0.50,
         -1.50, -1.00, -1.00,
         -0.50,  1.00,  0.50,
          1.00,  1.00,  1.50,
         -2.50, -2.00, -2.00,
         -1.50,  1.50,  1.50,
          2.00,  2.00,  2.50,
         -3.50, -3.00, -3.00,
         -2.50,  2.50,  2.50,
          3.00,  3.00,  3.50,
         -4.50, -4.00, -4.00,
         -3.50,  3.50,  3.50,
          4.00,  4.00,  4.50,
         -5.50, -5.00, -5.00,
         -4.50,  4.50,  4.50,
          5.00,  5.00,  5.50,
        };
    std::vector<float> weight = {
        -0.50,  0.00, -0.50,
        -0.50,  0.00, -0.50,
        -0.50,  0.00, -0.50,
         1.50,  1.00, -1.50,
         1.50,  1.00, -1.50,
         1.50,  1.00, -1.50,
        -2.50, -2.00, -2.50,
        -2.50, -2.00, -2.50,
        -2.50, -2.00, -2.50,

         3.50,  3.00,  3.50,
         3.50,  3.00,  3.50,
         3.50,  3.00,  3.50,
        -4.50, -4.00, -4.50,
        -4.50, -4.00, -4.50,
        -4.50, -4.00, -4.50,
        -5.50, -5.00,  5.50,
        -5.50, -5.00,  5.50,
        -5.50, -5.00,  5.50,
    };
    std::vector<float> bias = {
        -1.25, 1.25,
    };
    std::vector<float> golden = {
        -6.25, 27.25,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size() * sizeof(float)));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias.data(), bias.size() * sizeof(float)));

    std::array<uint32_t, 2> dilations = {1,1};
    std::array<uint32_t, 2> strides = {1,1};
    auto op = graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
        tim::vx::PadType::VALID, strides, dilations, 2);
    (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(GroupedConv2d, shape_3_3_6_1_uint8_group_6_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3,3,6,1});
    tim::vx::ShapeType weight_shape({2,2,1,6});
    tim::vx::ShapeType bias_shape({6});
    tim::vx::ShapeType output_shape({2,2,6,1});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 10);
    tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 9);
    tim::vx::Quantization bias_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 0.25, 85);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT,
                            input_quant);
    tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8,
                            weight_shape, tim::vx::TensorAttribute::CONSTANT,
                            weight_quant);
    tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32,
                            bias_shape, tim::vx::TensorAttribute::CONSTANT,
                            bias_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT,
                            output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec);
    auto bias_tensor = graph->CreateTensor(bias_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
         9,  9,  9,
        10, 12, 10,
        11, 11, 11,
         7,  8,  8,
         9, 12, 11,
        12, 12, 13,
         5,  6,  6,
         7, 13, 13,
        14, 14, 15,
         3,  4,  4,
         5, 15, 15,
        16, 16, 17,
         1,  2,  2,
         3, 17, 17,
        18, 18, 19,
         3,  0,  0,
         1, 19, 19,
        16,  4,  3,
        };
    std::vector<uint8_t> weight = {
         8,  9,
         8,  9,
        12, 11,
        12, 11,
         4,  5,
         4,  5,
        16, 15,
        16, 15,
         0, 17,
         0, 17,
         6,  5,
         6, 13,
    };
    std::vector<int32_t> bias = {
        -24,-20,-16, 16, -4, 20,
    };
    std::vector<uint8_t> golden = {
         62, 62,
         60, 60,
         53, 62,
         75, 74,
        113, 74,
         33, 44,
         11, 94,
        179,150,
        217, 90,
         73,  0,
        229,108,
        111,126,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size()));
    EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias.data(), bias.size()));

    std::array<uint32_t, 2> dilations = {1,1};
    std::array<uint32_t, 2> strides = {2,2};
    std::array<uint32_t, 4> pad = {0,1,0,1};
    auto op = graph->CreateOperation<tim::vx::ops::GroupedConv2d>(pad, strides, dilations, 6);
    (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}