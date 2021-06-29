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
#include <string>
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/moments.h"

#include "gtest/gtest.h"

namespace {
template<typename T>
::testing::AssertionResult ArraysMatch(const std::vector<T>& expected,
                                       const std::vector<T>& actual,
                                       T abs_error,
                                       const std::string& msg){
    for (size_t i = 0; i < expected.size(); ++i){
        EXPECT_NEAR(expected[i], actual[i], abs_error) << msg << " at index:" << i;
    }

    return ::testing::AssertionSuccess();
}
}

TEST(Moments, shape_6_3_1_float_axes_0_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({6, 3, 1});
    tim::vx::ShapeType output_shape({1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto mean = graph->CreateTensor(output_spec);
    auto variance = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -2, 0, 2,
        -3, 0, 3,
        -4, 0, 4,
        -5, 0, 5,
        -6, 0, 6,
        -7, 0, 7 };

    std::vector<float> mean_golden = {
        0
    };

    std::vector<float> variance_golden = {
        15.444444
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    std::vector<int32_t> axes = { 0, 1 };
    auto op = graph->CreateOperation<tim::vx::ops::Moments>(axes);
    (*op).BindInputs({input_tensor}).BindOutputs({mean, variance});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> mean_output(mean_golden);
    std::vector<float> variance_output(variance_golden);
    EXPECT_TRUE(mean->CopyDataFromTensor(mean_output.data()));
    EXPECT_TRUE(variance->CopyDataFromTensor(variance_output.data()));
    EXPECT_TRUE(ArraysMatch(mean_golden, mean_output, 1e-5f, "mean output"));
    EXPECT_TRUE(ArraysMatch(variance_golden, variance_output, 1e-5f, "variance output"));
}

TEST(Moments, shape_3_6_1_float_axes_1_keepdims) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3, 6, 1});
    tim::vx::ShapeType output_shape({3,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec mean_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec variance_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto mean = graph->CreateTensor(mean_spec);
    auto variance = graph->CreateTensor(variance_spec);

    std::vector<float> in_data = {
        -2, 0, 2,
        -3, 0, 3,
        -4, 0, 4,
        -5, 0, 5,
        -6, 0, 6,
        -7, 0, 7 
    };

    std::vector<float> mean_golden = {
        -4.5, 0, 4.5
    };

    std::vector<float> variance_golden = {
        2.916666, 0, 2.916666
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));

    std::vector<int32_t> axes = { 1 };
    auto op = graph->CreateOperation<tim::vx::ops::Moments>(axes, true);
    (*op).BindInputs({input_tensor}).BindOutputs({mean, variance});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> mean_output(mean_golden);
    std::vector<float> variance_output(variance_golden);
    EXPECT_TRUE(mean->CopyDataFromTensor(mean_output.data()));
    EXPECT_TRUE(variance->CopyDataFromTensor(variance_output.data()));
    EXPECT_TRUE(ArraysMatch(mean_golden, mean_output, 1e-5f, "mean output"));
    EXPECT_TRUE(ArraysMatch(variance_golden, variance_output, 1e-5f, "variance output"));
}

#if 0
// TODO: Support uint8
TEST(Moments, shape_3_6_1_uint8_axes_1_keepdims) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3, 6, 1});
    tim::vx::ShapeType output_shape({3,1,1});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1, 7);
    tim::vx::Quantization mean_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 9);
    tim::vx::Quantization variance_quant(tim::vx::QuantType::ASYMMETRIC, 2.916666, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec mean_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, mean_quant);
    tim::vx::TensorSpec variance_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, variance_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto mean = graph->CreateTensor(mean_spec);
    auto variance = graph->CreateTensor(variance_spec);

    std::vector<uint8_t> in_data = {
        5,  7,  9,
        4,  7, 10,
        3,  7, 11,
        2,  7, 12,
        1,  7, 13,
        0,  7, 14,
    };

    std::vector<uint8_t> mean_golden = {
        0, 9, 18
    };

    std::vector<uint8_t> variance_golden = {
        1, 0, 1
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

    std::vector<int32_t> axes = { 1 };
    auto op = graph->CreateOperation<tim::vx::ops::Moments>(axes, true);
    (*op).BindInputs({input_tensor}).BindOutputs({mean, variance});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> mean_output(mean_golden);
    std::vector<uint8_t> variance_output(variance_golden);
    EXPECT_TRUE(mean->CopyDataFromTensor(mean_output.data()));
    EXPECT_TRUE(variance->CopyDataFromTensor(variance_output.data()));
    EXPECT_EQ(mean_golden, mean_output);
    EXPECT_EQ(variance_golden, variance_output);
}
#endif
