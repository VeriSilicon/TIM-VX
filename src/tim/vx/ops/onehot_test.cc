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
#include "tim/vx/ops/onehot.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(OneHot, shape_3_out_flaot_depth_3) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            {3, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<float> golden = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(9);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int32_depth_3) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {3, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<int32_t> golden = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(9);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int8_depth_3) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8,
                            {3, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<int8_t> golden = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int8_t> output(9);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_uint8_depth_3) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            {3, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<uint8_t> golden = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(9);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int32_depth_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 1;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {3, 1}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<int32_t> golden = {1, 0, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(3);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int32_depth_4) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 4;

    tim::vx::ShapeType input_shape({3});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {3, 4}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 1, 2};

    std::vector<int32_t> golden = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(12);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int32_depth_3_on_6_off_N1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;
    float on = 6;
    float off = -1;

    tim::vx::ShapeType input_shape({4});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {4, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 2, -1, 1};

    std::vector<int32_t> golden = {6, -1, -1, -1, -1, 6, -1, -1, -1, -1, 6, -1};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth, on, off);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(12);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_3_out_int32_depth_3_on_5_off_0_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;
    float on = 5;
    float off = 0;
    int32_t axis = 1;

    tim::vx::ShapeType input_shape({4});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {4, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 2, -1, 1};

    std::vector<int32_t> golden = {5, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth, on, off, axis);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(12);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(OneHot, shape_2_2_out_int32_depth_3_on_2_off_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    int32_t depth = 3;
    float on = 2;
    float off = 0;
    int32_t axis = 0;

    tim::vx::ShapeType input_shape({2, 2});//AKA: indices
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            {2, 2, 3}, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> input_data = {0, 2, 1, -1};

    std::vector<int32_t> golden = {2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));

    auto op = graph->CreateOperation<tim::vx::ops::OneHot>(depth, on, off, axis);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(12);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}