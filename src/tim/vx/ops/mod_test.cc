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
#include "tim/vx/ops/mod.h"

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"
#include "test_utils.h"
#include "gtest/gtest.h"

#ifdef VSI_FEAT_OP_MOD

TEST(Mod, shape_2_2_3_1_fp32_fmod_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3, 1});
    tim::vx::ShapeType output_shape({3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = {
        3, 8, 7 };
    std::vector<float> in_data_y = {
        1, 3, 2 };
    std::vector<float> golden = {
        0, 2, 1};

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    //fmod = 0 
    auto op = graph->CreateOperation<tim::vx::ops::Mod>(0);
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(3);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
TEST(Mod, shape_2_2_3_1_fp32_fmod_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3, 1});
    tim::vx::ShapeType output_shape({3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = {
        3.9, 10.0, 10.9 };
    std::vector<float> in_data_y = {
        1.3, 4.4, 3.9 };
    std::vector<float> golden = {
        0.0, 1.2, 3.1};

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    //fmod = 1 
    auto op = graph->CreateOperation<tim::vx::ops::Mod>(1);
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(3);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}
TEST(Mod, shape_2_2_3_1_fp32_Broadcast) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input1_shape({3, 1});
    tim::vx::ShapeType input2_shape({1, 1});
    tim::vx::ShapeType output_shape({3, 1});
    tim::vx::TensorSpec input1_spec(tim::vx::DataType::FLOAT32,
                            input1_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec input2_spec(tim::vx::DataType::FLOAT32,
                            input2_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input1_spec);
    auto input_tensor_y = graph->CreateTensor(input2_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = {
        4, 11, 7};
    std::vector<float> in_data_y = {
        3};
    std::vector<float> golden = {
        1, 2, 1};

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    // fmod is set to be 0 default if not specified;
    auto op = graph->CreateOperation<tim::vx::ops::Mod>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(3);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
#endif  //(VSI_FEAT_OP_MOD)
