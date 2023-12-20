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
#include "tim/vx/ops/layernormalization.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(LayerNorm, axis_0_shape_3_6_1_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    float tolerance = ctx->hasSP() ? 0.01 : 1e-5f;

    tim::vx::ShapeType io_shape({3, 6, 1});
    tim::vx::ShapeType param_shape({6});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto gamma_tensor = graph->CreateTensor(param_spec);
    auto beta_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -2, 0, 2,
        -3, 0, 3,
        -4, 0, 4,
        -5, 0, 5,
        -6, 0, 6,
        -7, 0, 7 };
    std::vector<float> gamma = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    std::vector<float> beta = {
        .0f, .0f, .0f,
        .0f, .0f, .0f
    };
    std::vector<float> golden = {
        -1.22474f, 0, 1.22474f,
        -1.22474f, 0, 1.22474f,
        -1.22474f, 0, 1.22474f,
        -1.22474f, 0, 1.22474f,
        -1.22474f, 0, 1.22474f,
        -1.22474f, 0, 1.22474f,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(gamma_tensor->CopyDataToTensor(gamma.data(), gamma.size() * sizeof(float)));
    EXPECT_TRUE(beta_tensor->CopyDataToTensor(beta.data(), beta.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LayerNormalization>(0, 2e-5f);
    (*op).BindInputs({input_tensor, beta_tensor, gamma_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(18);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}

TEST(LayerNorm, axis_0_shape_2_3_6_1_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    float tolerance = ctx->hasSP() ? 0.01 : 1e-5f;

    tim::vx::ShapeType io_shape({2, 3, 6, 1});
    tim::vx::ShapeType param_shape({6});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto gamma_tensor = graph->CreateTensor(param_spec);
    auto beta_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -2, 2, -2, 2, -2, 2,
        -3, 3, -3, 3, -3, 3,
        -4, 4, -4, 4, -4, 4,
        -5, 5, -5, 5, -5, 5,
        -6, 6, -6, 6, -6, 6,
        -7, 7, -7, 7, -7, 7
        };
    std::vector<float> gamma = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    std::vector<float> beta = {
        .0f, .0f, .0f,
        .0f, .0f, .0f
    };
    std::vector<float> golden = {
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
        -1.f, 1.f, -1.f, 1.f, -1.f, 1.f,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(gamma_tensor->CopyDataToTensor(gamma.data(), gamma.size() * sizeof(float)));
    EXPECT_TRUE(beta_tensor->CopyDataToTensor(beta.data(), beta.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LayerNormalization>(0, 2e-5f);
    (*op).BindInputs({input_tensor, beta_tensor, gamma_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(36);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}

TEST(LayerNorm, axis_2_shape_4_2_3_1_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    float tolerance = ctx->hasSP() ? 0.01 : 1e-5f;

    tim::vx::ShapeType io_shape({4, 2, 3, 1});
    tim::vx::ShapeType param_shape({1,1,3,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto gamma_tensor = graph->CreateTensor(param_spec);
    auto beta_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24};
    std::vector<float> gamma = {
        1.0f, 1.0f, 1.0f
    };
    std::vector<float> beta = {
        .0f, .0f, .0f
    };
    std::vector<float> golden = {
        -1.22473, -1.22473, -1.22473, -1.22473,
        -1.22473, -1.22473, -1.22473, -1.22473,
         0, 0, 0, 0,
         0, 0, 0, 0,
         1.22473, 1.22473, 1.22473, 1.22473,
         1.22473, 1.22473, 1.22473, 1.22473
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(gamma_tensor->CopyDataToTensor(gamma.data(), gamma.size() * sizeof(float)));
    EXPECT_TRUE(beta_tensor->CopyDataToTensor(beta.data(), beta.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LayerNormalization>(2, 0.001);
    (*op).BindInputs({input_tensor, beta_tensor, gamma_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(24);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}

#if 0
// Fail case
TEST(LayerNorm, axis_0_shape_3_6_1_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 6, 1});
    tim::vx::ShapeType param_shape({6});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1, 7);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1.22474f, 1);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::INPUT, input_quant);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto gamma_tensor = graph->CreateTensor(param_spec);
    auto beta_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        5, 7, 9,
        4, 7, 10,
        3, 7, 11,
        2, 7, 12,
        1, 7, 13,
        0, 7, 14 };
    std::vector<float> gamma = {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    };
    std::vector<float> beta = {
        .0f, .0f, .0f,
        .0f, .0f, .0f
    };
    std::vector<float> golden = {
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
        0, 1, 2,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(gamma_tensor->CopyDataToTensor(gamma.data(), gamma.size() * sizeof(float)));
    EXPECT_TRUE(beta_tensor->CopyDataToTensor(beta.data(), beta.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::LayerNormalization>(0, 2e-5f);
    (*op).BindInputs({input_tensor, beta_tensor, gamma_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(18);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
#endif
