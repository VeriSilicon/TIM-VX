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
#include "tim/vx/ops/resize1d.h"

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

TEST(Resize1d, shape_4_2_1_float_nearest_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({4, 2, 1});
    tim::vx::ShapeType output_shape({2, 2, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1.f, 2.f, 3.f, 4.f,
        5.f, 6.f, 7.f, 8.f,
        };
    std::vector<float> golden = {
        1.f, 3.f,
        5.f, 7.f,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Resize1d>(
        tim::vx::ResizeType::NEAREST_NEIGHBOR, 0.6, false, false, 0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Resize1d, shape_4_2_1_uint8_nearest_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({4, 2, 1});
    tim::vx::ShapeType output_shape({2, 2, 1});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        };
    std::vector<uint8_t> golden = {
        1, 3,
        5, 7,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::Resize1d>(
        tim::vx::ResizeType::NEAREST_NEIGHBOR, 0.6, false, false, 0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Resize1d, shape_5_1_1_float_bilinear_align_corners_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({5, 1, 1});
    tim::vx::ShapeType output_shape({7, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1.f, 2.f, 3.f, 4.f, 5.f,
        };
    std::vector<float> golden = {
        1.f, 1.66666f, 2.33333f, 3.f,
        3.66666, 4.33333f, 5.f
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Resize1d>(
        tim::vx::ResizeType::BILINEAR, 0, true, false, 7);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}
