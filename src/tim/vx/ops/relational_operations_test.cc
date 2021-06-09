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
#include "tim/vx/ops/relational_operations.h"

#include "gtest/gtest.h"

TEST(Equal, shape_1_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            io_shape, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data1 = { 255 };
    std::vector<uint8_t> in_data2 = { 0 };

    std::vector<uint8_t> golden = {0};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()));

    auto op = graph->CreateOperation<tim::vx::ops::Equal>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(1, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(NotEqual, shape_5_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> in_data2 = { -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<uint8_t> golden = {1, 1, 1, 0, 0};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::NotEqual>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Less, shape_5_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,5});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = { 0.1, 0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> in_data2 = { -1, -1, 0.2, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<uint8_t> golden = {0, 0, 1, 0, 0};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::Less>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(GreaterOrEqual, shape_5_2_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5,2,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity(),
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> in_data2 = {
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity(),
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<uint8_t> golden = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::GreaterOrEqual>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(10, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Greater, shape_5_2_1_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5,2,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity(),
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> in_data2 = {
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity(),
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<uint8_t> golden = {0, 1, 0, 0, 0, 0, 1, 0, 0, 0};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::Greater>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(10, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(LessOrEqual, shape_1_5_2_1_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,5,2,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity(),
        -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> in_data2 = {
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity(),
        -2, -1, 0.2, 0.55, std::numeric_limits<float>::infinity() };

    std::vector<uint8_t> golden = {1, 0, 1, 1, 1, 1, 0, 1, 1, 1};

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(in_data1.data(), in_data1.size()*4));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(in_data2.data(), in_data2.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::LessOrEqual>();
    (*op).BindInputs({input_tensor1, input_tensor2}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    //Not using vector<bool> because it uses a bitfield representation internally
    //and it's cumbersome to copy tensor data to it.
    std::vector<uint8_t> output(10, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

