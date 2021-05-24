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
#include "tim/vx/ops/elementwise.h"

#include "gtest/gtest.h"

TEST(FloorDiv, shape_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = { 1 };
    std::vector<float> in_data_y = { 0 };
    std::vector<float> golden = { std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(1);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(FloorDiv, shape_5_1_broadcast_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({5, 1});
    tim::vx::ShapeType in_shape_y({1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::FLOAT32,
                            in_shape_x, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::FLOAT32,
                            in_shape_y, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = { 1, 3, -2, 0, 99 };
    std::vector<float> in_data_y = { 2 };
    std::vector<float> golden = { 0, 1, -1, 0, 49 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}


TEST(FloorDiv, shape_5_1_broadcast_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({1});
    tim::vx::ShapeType in_shape_y({5, 1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::UINT8,
                            in_shape_x, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::UINT8,
                            in_shape_y, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, quant_out);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data_x = { 255 };
    std::vector<uint8_t> in_data_y = { 1, 3, 2, 0, 255 };
    std::vector<uint8_t> golden = { 255, 170, 254, 255, 2 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
