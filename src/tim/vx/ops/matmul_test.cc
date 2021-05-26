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
#include "tim/vx/ops/matmul.h"

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

TEST(Matmul, shape_2_6_shape_6_2_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType a_shape({6, 2});
    tim::vx::ShapeType b_shape({2, 6});
    tim::vx::ShapeType out_shape({2, 2});
    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                    a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::FLOAT32,
                    b_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> a_data = {
        1, 2, 3, 4, 5, 6,
        -1, -2, -3, -4, -5, -6
    };
    std::vector<float> b_data = {
        6, 5,
        4, 3,
        2, 1,
        -6, -5,
        -4, -3,
        -2, -1
    };
    std::vector<float> golden = {
        -36, -27,
        36, 27
    };

    EXPECT_TRUE(a_tensor->CopyDataToTensor(a_data.data(), a_data.size() * sizeof(float)));
    EXPECT_TRUE(b_tensor->CopyDataToTensor(b_data.data(), b_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Matmul>();
    (*op).BindInputs({a_tensor, b_tensor}).BindOutputs({out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Matmul, shape_2_3_2_shape_2_3_2_float_transpose_b) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType a_shape({2, 3, 2});
    tim::vx::ShapeType b_shape({2, 3, 2});
    tim::vx::ShapeType out_shape({3, 3, 2});
    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                    a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::FLOAT32,
                    b_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> a_data = {
        1, 2,
        3, 4,
        5, 6,
        -1, -2,
        -3, -4,
        -5, -6
    };
    std::vector<float> b_data = {
        6, 5,
        4, 3,
        2, 1,
        -6, -5,
        -4, -3,
        -2, -1
    };
    std::vector<float> golden = {
        16, 10,  4,
        38, 24, 10,
        60, 38, 16,
        16, 10,  4,
        38, 24, 10,
        60, 38, 16,
    };

    EXPECT_TRUE(a_tensor->CopyDataToTensor(a_data.data(), a_data.size() * sizeof(float)));
    EXPECT_TRUE(b_tensor->CopyDataToTensor(b_data.data(), b_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Matmul>(false, true);
    (*op).BindInputs({a_tensor, b_tensor}).BindOutputs({out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Matmul, shape_2_3_2_shape_2_3_2_uint8_transpose_a) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType a_shape({2, 3, 2});
    tim::vx::ShapeType b_shape({2, 3, 2});
    tim::vx::ShapeType out_shape({2, 2, 2});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1, 6);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec a_spec(tim::vx::DataType::UINT8,
                    a_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::UINT8,
                    b_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::UINT8,
                    out_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<uint8_t> a_data = {
         7,  8,
         9, 10,
        11, 12,
         5,  4,
         3,  2,
         1,  0,
    };
    std::vector<uint8_t> b_data = {
        12, 11,
        10,  9,
         8,  7,
         0,  1,
         2,  3,
         4,  5,
    };
    std::vector<uint8_t> golden = {
        28, 19,
        40, 28,
        28, 19,
        40, 28,
    };

    EXPECT_TRUE(a_tensor->CopyDataToTensor(a_data.data(), a_data.size()));
    EXPECT_TRUE(b_tensor->CopyDataToTensor(b_data.data(), b_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::Matmul>(true);
    (*op).BindInputs({a_tensor, b_tensor}).BindOutputs({out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}