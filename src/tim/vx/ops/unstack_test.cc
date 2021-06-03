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
#include "tim/vx/ops/unstack.h"

#include "gtest/gtest.h"

TEST(Unstack, shape_4_3_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({4,3});
    tim::vx::ShapeType output_shape({3});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output1_tensor = graph->CreateTensor(output_spec);
    auto output2_tensor = graph->CreateTensor(output_spec);
    auto output3_tensor = graph->CreateTensor(output_spec);
    auto output4_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1,2,3,4,
        5,6,7,8,
        9,10,11,12,
        };
    std::vector<float> golden1 = {
        1,5,9
    };
    std::vector<float> golden2 = {
        2,6,10
    };
    std::vector<float> golden3 = {
        3,7,11
    };
    std::vector<float> golden4 = {
        4,8,12
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Unstack>(0, 4);
    (*op).BindInputs({input_tensor}).BindOutputs(
        {output1_tensor, output2_tensor, output3_tensor, output4_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output1(golden1.size());
    std::vector<float> output2(golden2.size());
    std::vector<float> output3(golden3.size());
    std::vector<float> output4(golden4.size());
    EXPECT_TRUE(output1_tensor->CopyDataFromTensor(output1.data()));
    EXPECT_TRUE(output2_tensor->CopyDataFromTensor(output2.data()));
    EXPECT_TRUE(output3_tensor->CopyDataFromTensor(output3.data()));
    EXPECT_TRUE(output4_tensor->CopyDataFromTensor(output4.data()));
    EXPECT_EQ(golden1, output1);
    EXPECT_EQ(golden2, output2);
    EXPECT_EQ(golden3, output3);
    EXPECT_EQ(golden4, output4);
}

TEST(Unstack, shape_4_3_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({4,3});
    tim::vx::ShapeType output_shape({4});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
        input_shape, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
        output_shape, tim::vx::TensorAttribute::OUTPUT, quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output1_tensor = graph->CreateTensor(output_spec);
    auto output2_tensor = graph->CreateTensor(output_spec);
    auto output3_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        2,4,6,8,
        10,12,14,16,
        18,20,22,24,
        };
    std::vector<uint8_t> golden1 = {
        2,4,6,8
    };
    std::vector<uint8_t> golden2 = {
        10,12,14,16,
    };
    std::vector<uint8_t> golden3 = {
        18,20,22,24,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Unstack>(1, 3);
    (*op).BindInputs({input_tensor}).BindOutputs(
        {output1_tensor, output2_tensor, output3_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output1(golden1.size());
    std::vector<uint8_t> output2(golden2.size());
    std::vector<uint8_t> output3(golden3.size());
    EXPECT_TRUE(output1_tensor->CopyDataFromTensor(output1.data()));
    EXPECT_TRUE(output2_tensor->CopyDataFromTensor(output2.data()));
    EXPECT_TRUE(output3_tensor->CopyDataFromTensor(output3.data()));
    EXPECT_EQ(golden1, output1);
    EXPECT_EQ(golden2, output2);
    EXPECT_EQ(golden3, output3);
}
