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
#include "tim/vx/ops/reorg.h"

#include "gtest/gtest.h"

// FIXME (KC) : There seems to be a limitation that Channel needs to be >= 4,
//              also stride other than 2 is not tested
TEST(Reorg, shape_4_4_4_1_u8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType i_shape({4, 4, 4, 1});
    tim::vx::ShapeType o_shape({2, 2, 16, 1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                        i_shape, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                        o_shape, tim::vx::TensorAttribute::OUTPUT, quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };
    std::vector<uint8_t> golden = {
        0, 2, 4, 6, 16, 18, 20, 22, 0, 2, 4, 6, 16, 18, 20, 22,
        1, 3, 5, 7, 17, 19, 21, 23, 1, 3, 5, 7, 17, 19, 21, 23,
        8, 10, 12, 14, 24, 26, 28, 30, 8, 10, 12, 14, 24, 26, 28, 30,
        9, 11, 13, 15, 25, 27, 29, 31, 9, 11, 13, 15, 25, 27, 29, 31
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::Reorg>(2);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(64, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Reorg, shape_4_4_4_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType i_shape({4, 4, 4, 1});
    tim::vx::ShapeType o_shape({2, 2, 16, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            i_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            o_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };
    std::vector<float> golden = {
        0, 2, 4, 6, 16, 18, 20, 22, 0, 2, 4, 6, 16, 18, 20, 22,
        1, 3, 5, 7, 17, 19, 21, 23, 1, 3, 5, 7, 17, 19, 21, 23,
        8, 10, 12, 14, 24, 26, 28, 30, 8, 10, 12, 14, 24, 26, 28, 30,
        9, 11, 13, 15, 25, 27, 29, 31, 9, 11, 13, 15, 25, 27, 29, 31
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::Reorg>(2);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(64, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

