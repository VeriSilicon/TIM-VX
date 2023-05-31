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
#include "tim/vx/ops/groupedconv1d.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(GroupedConv1d, shape_6_2_1_float_ksize_6_stride_1_group_2_no_bias_wcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({6, 2, 1});
    tim::vx::ShapeType param_shape({6, 1, 2});
    tim::vx::ShapeType out_shape({1, 2, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -1,    0, 1, -1.5, 0.5, 1.5,
        -2, -0.5, 2, -2.5,   0, 2.5,
        };
    std::vector<float> weight = {
        -3,   -2, -1.5, 1.5, 2,   3,
        -2.5, -2, -1.5, 1.5, 2, 2.5,
    };
    std::vector<float> golden = {
        4.75, 5.5,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::GroupedConv1d>(tim::vx::PadType::VALID, 1, 1, 2);
    (*op).BindInputs({input_tensor, weight_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(GroupedConv1d, shape_6_2_1_float_ksize_6_stride_1_group_2_no_bias_wcn_PaddingTest) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({2, 4, 1});
    tim::vx::ShapeType param_shape({3, 2, 4});
    tim::vx::ShapeType out_shape({2, 4, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(param_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -1, 0, 1, -1.5, 0.5, 1.5, 1, 1
        };
    std::vector<float> weight = {
        -3,   -2, -1.5, 1.5, 2,   3,
        -2.5, -2, -1.5, 1.5, 2, 2.5,
        -1,    0,    1, -1.5, 0.5, 1.5,
        -1.5, 1.5, 2, -1, 0, 1,
    };
    std::vector<float> golden = {
        1.5, -2.25, 1, -2.25, -1.5, -3, 0.5, -3.25
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weight_tensor->CopyDataToTensor(weight.data(), weight.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::GroupedConv1d>(tim::vx::PadType::VALID, 1, 1, 2);
    (*op).BindInputs({input_tensor, weight_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    // EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
    EXPECT_EQ(golden, output);
}
