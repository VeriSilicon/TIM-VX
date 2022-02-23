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
#include "tim/vx/ops/stack.h"

#include "gtest/gtest.h"

TEST(Stack, shape_2_3_axis_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1,4,
        2,5,
        3,6,

        1,4,
        2,5,
        3,6
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(2, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Stack, shape_2_3_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1,4,
        1,4,
        2,5,

        2,5,
        3,6,
        3,6,
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(1, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Stack, shape_2_3_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1, 1,
        4, 4,
        2, 2,

        5, 5,
        3, 3,
        6, 6
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(0, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
