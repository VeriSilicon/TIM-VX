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
#ifdef VSI_FEAT_OP_MAXPOOLWITHARGMAX
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/maxpoolgrad.h"
#include "tim/vx/ops/scatternd.h"
#include "tim/vx/ops/reshape.h"

#include "gtest/gtest.h"

TEST(Fuse_MaxpoolGrad, without_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({6, 4, 1, 1});
    tim::vx::ShapeType updates_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            updates_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        7, 2, 5, 3, 10, 2,
        3, 8, 9, 3, 4, 2,
        1, 5, 7, 5, 6, 1,
        0, 6, 2, 7, 2, 8};
    std::vector<float> updates_data = {
        2, 6,
        3, 1
        };
    std::vector<float> golden = {
        0, 0, 0, 0, 6, 0,
        0, 0, 2, 0, 0, 0,
        0, 0, 3, 0, 0, 0,
        0, 0, 0, 0, 0, 1};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(updates_data.data(), updates_data.size() * sizeof(float)));

    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {3, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output_values(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}

TEST(Fuse_MaxpoolGrad, with_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 4, 1, 1});
    tim::vx::ShapeType updates_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            updates_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(input_spec);

    std::vector<float> in_data = {
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2};
    std::vector<float> updates_data = {
        2, 6,
        3, 1
        };
    std::vector<float> golden = {
        0, 0, 0, 0, 0,
        0, 0, 8, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 1, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(updates_data.data(), updates_data.size() * sizeof(float)));


    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    
    std::vector<float> output_values(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}

TEST(Fuse_MaxpoolGrad, with_overlay_multi_channel_multi_batch) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 4, 2, 2});
    tim::vx::ShapeType updates_shape({2, 2, 2, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            updates_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(input_spec);

    std::vector<float> in_data = {
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2,
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2,
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2,
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2};
    std::vector<float> updates_data = {
        2, 6,
        3, 1,
        2, 6,
        3, 1,
        2, 6,
        3, 1,
        2, 6,
        3, 1,
        };
    std::vector<float> golden = {
        0, 0, 0, 0, 0,
        0, 0, 8, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 8, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 8, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0,
        0, 0, 8, 0, 0,
        0, 0, 3, 0, 0,
        0, 0, 0, 1, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(updates_data.data(), updates_data.size() * sizeof(float)));


    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    
    std::vector<float> output_values(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}
#endif