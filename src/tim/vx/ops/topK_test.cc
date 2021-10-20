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
#include "tim/vx/ops/topK.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(TopK, shape_12_1_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 3;
    tim::vx::ShapeType in_shape({12, 1});
    tim::vx::ShapeType out_shape_values({k, 1});
    tim::vx::ShapeType out_shape_indices({k, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::INT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<int32_t> in_data = {
        1, 2, 98, 1, 1, 99, 3, 1, 3, 96, 4, 1};
    std::vector<int32_t> golden_values = {
        99, 98, 96};  //correct answer
    std::vector<int32_t> golden_indices = {
        5, 2, 9};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_EQ(golden_indices, output_indices);
}

TEST(TopK, shape_14_1_int32_equal_elements) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 5;
    tim::vx::ShapeType in_shape({14, 1});
    tim::vx::ShapeType out_shape_values({k, 1});
    tim::vx::ShapeType out_shape_indices({k, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::INT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<int32_t> in_data = {
        1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
    std::vector<int32_t> golden_values = {
        1, 1, 1, 1, 1};  //correct answer
    std::vector<int32_t> golden_indices = {
        0, 1, 3, 5, 8};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_EQ(golden_indices, output_indices);
}

TEST(TopK, shape_5_2_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 4;
    tim::vx::ShapeType in_shape({5,2});
    tim::vx::ShapeType out_shape_values({k,2});
    tim::vx::ShapeType out_shape_indices({k,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::INT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<int32_t> in_data = {
        1,2,3,4,5,
        4,5,6,7,8};
    std::vector<int32_t> golden_values = {
        5,4,3,2,
        8,7,6,5};  //correct answer
    std::vector<int32_t> golden_indices = {
        4,3,2,1,
        4,3,2,1};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_EQ(golden_indices, output_indices);
}

TEST(TopK, shape_4_3_2_2_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 2;
    tim::vx::ShapeType in_shape({4,3,2,2}); //Reverse the layout of C language
    tim::vx::ShapeType out_shape_values({k,3,2,2});
    tim::vx::ShapeType out_shape_indices({k,3,2,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::INT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<int32_t> in_data = {
        5, 5, 1, 1,
        1, 6, 1, 1,
        1, 3, 1, 1,

        1, 2, 1, 1,
        2, 9, 2, 1,
        1, 8, 1, 1,

        5, 5, 1, 1,
        1, 6, 1, 1,
        1, 3, 1, 1,

        1, 2, 1, 1,
        2, 9, 2, 1,
        1, 8, 1, 1};
    std::vector<int32_t> golden_values = {
        5, 5,
        6, 1,
        3, 1,

        2, 1,
        9, 2,
        8, 1,

        5, 5,
        6, 1,
        3, 1,

        2, 1,
        9, 2,
        8, 1};  //correct answer
    std::vector<int32_t> golden_indices = {
        0, 1,
        1, 0,
        1, 0,

        1, 0,
        1, 0,
        1, 0,

        0, 1,
        1, 0,
        1, 0,

        1, 0,
        1, 0,
        1, 0};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_EQ(golden_indices, output_indices);
}

TEST(TopK, shape_2_2_2_2_2_2_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 1;
    tim::vx::ShapeType in_shape({2,2,2,2,2,2}); //Reverse the layout of C language
    tim::vx::ShapeType out_shape_values({k,2,2,2,2,2});
    tim::vx::ShapeType out_shape_indices({k,2,2,2,2,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::INT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<int32_t> in_data = {
        5, 5,
        1, 1,

        1, 6,
        1, 1,

        1, 3,
        1, 1,

        1, 2,
        1, 1,

        2, 9,
        2, 1,

        1, 8,
        1, 1,

        5, 5,
        1, 1,

        1, 6,
        1, 1,

        1, 3,
        1, 1,

        1, 2,
        1, 1,

        2, 9,
        2, 1,

        1, 8,
        1, 1,

        3, 4,
        5, 9,

        1, 6,
        5, 3,

        9, 4,
        0, 4,

        7, 6,
        9, 4};
    std::vector<int32_t> golden_values = {
        5,1,6,1,3,1,2,1,
        9,2,8,1,5,1,6,1,
        3,1,2,1,9,2,8,1,
        4,9,6,5,9,4,7,9};  //correct answer
    std::vector<int32_t> golden_indices = {
        0,0,1,0,1,0,1,0,
        1,0,1,0,0,0,1,0,
        1,0,1,0,1,0,1,0,
        1,1,1,0,0,1,0,0};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_EQ(golden_indices, output_indices);
}

TEST(TopK, shape_4_3_2_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    uint32_t k = 1;
    tim::vx::ShapeType in_shape({4,3,2});
    tim::vx::ShapeType out_shape_values({k,3,2});
    tim::vx::ShapeType out_shape_indices({k,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape_values, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::INT32,
                            out_shape_indices, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);

    std::vector<float> in_data = {
         0.44809645,  0.94678545,  1.0046302 , -1.3940862 ,
         1.2808152 , -0.49903643,  1.1129634 , -1.135716  ,
        -0.871886  , -0.38303357,  1.2514114 ,  0.8325187 ,

        -1.2743613 , -1.5972878 , -2.0661716 , -0.40234366,
        -0.16834415, -1.1711465 ,  0.435073  , -0.40024215,
         1.0241159 , -0.19233227, -0.06416418,  0.2981165 };
    std::vector<float> golden_values = {
         1.0046302 ,
         1.2808152,
         1.2514114,

        -0.40234366,
         0.435073  ,
         1.0241159 };  //correct answer
    std::vector<int32_t> golden_indices = {
        2,
        0,
        2,

        3,
        2,
        0};  //correct answer

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::TopK>(k);
    (*op).BindInput(input_tensor).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(golden_values.size());
    std::vector<int32_t> output_indices(golden_indices.size());

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(golden_values, output_values);
    EXPECT_TRUE(ArraysMatch(golden_values, output_values, 1e-5f));
    EXPECT_EQ(golden_indices, output_indices);
}
