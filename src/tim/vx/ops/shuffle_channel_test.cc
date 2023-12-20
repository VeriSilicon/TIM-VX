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
#include "tim/vx/ops/shuffle_channel.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(ShuffleChannel, shape_3_6_groupnum2_dim1_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 6});   //3 columns and 4 rows, w h c n
    tim::vx::ShapeType out_shape({3, 6});
    tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32,
                    in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto in_tensor = graph->CreateTensor(in_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18
    };
    std::vector<float> golden = {
        1, 2, 3,
        10, 11, 12,
        4, 5, 6,
        13, 14, 15,
        7, 8, 9,
        16, 17, 18
    };

    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ShuffleChannel>(2, 1);
    (*op).BindInput(in_tensor).BindOutput(out_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(ShuffleChannel, shape_4_2_2_groupnum2_dim0_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 2, 2});
    tim::vx::ShapeType out_shape({4, 2, 2});
    tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32,
                    in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto in_tensor = graph->CreateTensor(in_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    std::vector<float> golden = {
         1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16
    };

    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ShuffleChannel>(2, 0);
    (*op).BindInput(in_tensor).BindOutput(out_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(ShuffleChannel, shape_1_4_2_2_groupnum2_dim1_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({1, 4, 2, 2});
    tim::vx::ShapeType out_shape({1, 4, 2, 2});
    tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32,
                    in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto in_tensor = graph->CreateTensor(in_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    std::vector<float> golden = {
         1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16
    };

    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ShuffleChannel>(2, 1);
    (*op).BindInput(in_tensor).BindOutput(out_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(ShuffleChannel, shape_4_1_2_2_groupnum4_dim0_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 1, 2, 2});
    tim::vx::ShapeType out_shape({4, 1, 2, 2});
    tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32,
                    in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto in_tensor = graph->CreateTensor(in_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    std::vector<float> golden = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };

    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ShuffleChannel>(4, 0);
    (*op).BindInput(in_tensor).BindOutput(out_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(ShuffleChannel, shape_4_1_2_2_groupnum1_dim3_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 1, 2, 2});
    tim::vx::ShapeType out_shape({4, 1, 2, 2});
    tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32,
                    in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto in_tensor = graph->CreateTensor(in_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    std::vector<float> golden = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };

    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ShuffleChannel>(1, 3);
    (*op).BindInput(in_tensor).BindOutput(out_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}