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
#include "tim/vx/ops/logsoftmax.h"

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

TEST(LogSoftmax, shape_6_1_float_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({6, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 3, 4, 5, 6, 7
    };
    std::vector<float> golden = {
        -5.4562, -4.4562, -3.4562, -2.4562, -1.4562, -0.4562,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LogSoftmax>(0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(LogSoftmax, shape_3_6_1_float_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 6, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -2.0000,  0.0000,  2.0000,
        -3.0000,  0.0000,  3.0000,
        -4.0000,  0.0000,  4.0000,
        -5.0000,  0.0000,  5.0000,
        -6.0000,  0.0000,  6.0000,
        -7.0000,  0.0000,  7.0000,
    };
    std::vector<float> golden = {
        -0.4561933, -1.7917595, -5.4561934,
        -1.4561933, -1.7917595, -4.4561934,
        -2.4561934, -1.7917595, -3.4561934,
        -3.4561934, -1.7917595, -2.4561934,
        -4.4561934, -1.7917595, -1.4561933,
        -5.4561934, -1.7917595, -0.4561933,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LogSoftmax>(1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(LogSoftmax, shape_3_6_1_uint8_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 6, 1});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1, 2);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1.7917595, 2);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        0,  2,  4,
        0,  2,  4,
        0,  2,  4,
        0,  2,  4,
        0,  2,  4,
        0,  2,  4,
    };
    std::vector<uint8_t> golden = {
        1,  1,  1,
        1,  1,  1,
        1,  1,  1,
        1,  1,  1,
        1,  1,  1,
        1,  1,  1,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::LogSoftmax>(1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
