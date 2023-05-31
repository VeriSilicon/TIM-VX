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
#include "tim/vx/ops/maxpoolwithargmax2.h"
#include "tim/vx/ops/scatternd.h"
#include "tim/vx/ops/reshape.h"

#include "gtest/gtest.h"

#ifdef VSI_FEAT_OP_MAXPOOLWITHARGMAX

TEST(MaxpoolWithArgmax2, without_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({6, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<float> in_data = {
        7, 2, 5, 3, 10, 2,
        3, 8, 9, 3, 4, 2,
        1, 5, 7, 5, 6, 1,
        0, 6, 2, 7, 2, 8};
    std::vector<float> values_golden = {
        9, 10,
        7, 8 };
    std::vector<int32_t> indices_golden = {
        8, 4,
        14, 23 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {3, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(4);
    std::vector<int32_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}

TEST(MaxpoolWithArgmax2, with_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<float> in_data = {
        7, 2, 5, 3, 8,
        3, 8, 9, 3, 4,
        1, 5, 7, 5, 6,
        0, 6, 2, 10, 2};
    std::vector<float> values_golden = {
        9, 9,
        7, 10 };
    std::vector<int32_t> indices_golden = {
        7, 7,
        12, 18 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(4);
    std::vector<int32_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}

TEST(MaxpoolGrad, without_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({6, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
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

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {3, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    std::vector<uint32_t> shape = {4};
    tim::vx::TensorSpec input_spec_indices(tim::vx::DataType::INT32,
                            shape, tim::vx::TensorAttribute::TRANSIENT);
    auto input_tensor_indices = graph->CreateTensor(input_spec_indices);

    auto op1 = graph->CreateOperation<tim::vx::ops::Reshape>(shape);
    (*op1).BindInputs({output_tensor_indices}).BindOutputs({input_tensor_indices});

    std::vector<uint32_t> out2_shape = {24};
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output2_spec(tim::vx::DataType::FLOAT32,
                            out2_shape, tim::vx::TensorAttribute::TRANSIENT);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output2_tensor = graph->CreateTensor(output2_spec);
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size() * 4));

    auto op2 = graph->CreateOperation<tim::vx::ops::ScatterND>(out2_shape);
    (*op2).BindInputs({input_tensor_indices, updates_tensor}).BindOutputs({output2_tensor});

    auto op3 = graph->CreateOperation<tim::vx::ops::Reshape>(in_shape);
    (*op3).BindInputs({output2_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(24);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}

TEST(MaxpoolGrad, with_overlay) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor = graph->CreateTensor(output_spec);

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

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    std::vector<uint32_t> shape = {4};
    tim::vx::TensorSpec input_spec_indices(tim::vx::DataType::INT32,
                            shape, tim::vx::TensorAttribute::TRANSIENT);
    auto input_tensor_indices = graph->CreateTensor(input_spec_indices);

    auto op1 = graph->CreateOperation<tim::vx::ops::Reshape>(shape);
    (*op1).BindInputs({output_tensor_indices}).BindOutputs({input_tensor_indices});

    std::vector<uint32_t> out2_shape = {20};
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output2_spec(tim::vx::DataType::FLOAT32,
                            out2_shape, tim::vx::TensorAttribute::TRANSIENT);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output2_tensor = graph->CreateTensor(output2_spec);
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size() * 4));

    auto op2 = graph->CreateOperation<tim::vx::ops::ScatterND>(out2_shape);
    (*op2).BindInputs({input_tensor_indices, updates_tensor}).BindOutputs({output2_tensor});

    auto op3 = graph->CreateOperation<tim::vx::ops::Reshape>(in_shape);
    (*op3).BindInputs({output2_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(20);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}

TEST(MaxpoolGrad, with_overlay_multi_channel_multi_batch) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 4, 2, 2});
    tim::vx::ShapeType out_shape({2, 2, 2, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor = graph->CreateTensor(output_spec);

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

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
    std::array<uint32_t, 2> ksize = {3, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax2>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    std::vector<uint32_t> shape = {16};
    tim::vx::TensorSpec input_spec_indices(tim::vx::DataType::INT32,
                            shape, tim::vx::TensorAttribute::TRANSIENT);
    auto input_tensor_indices = graph->CreateTensor(input_spec_indices);

    auto op1 = graph->CreateOperation<tim::vx::ops::Reshape>(shape);
    (*op1).BindInputs({output_tensor_indices}).BindOutputs({input_tensor_indices});

    std::vector<uint32_t> out2_shape = {80};
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output2_spec(tim::vx::DataType::FLOAT32,
                            out2_shape, tim::vx::TensorAttribute::TRANSIENT);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output2_tensor = graph->CreateTensor(output2_spec);
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size() * 4));

    auto op2 = graph->CreateOperation<tim::vx::ops::ScatterND>(out2_shape);
    (*op2).BindInputs({input_tensor_indices, updates_tensor}).BindOutputs({output2_tensor});

    auto op3 = graph->CreateOperation<tim::vx::ops::Reshape>(in_shape);
    (*op3).BindInputs({output2_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(80);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(golden, output_values);
}

#endif //(VSI_FEAT_OP_MAXPOOLWITHARGMAX)