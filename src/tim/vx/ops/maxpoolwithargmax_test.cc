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
#include "tim/vx/ops/maxpoolwithargmax.h"

#include "gtest/gtest.h"

TEST(MaxpoolWithArgmax, shape_3_3_1_fp32_kernel_2_stride_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 3, 1});
    tim::vx::ShapeType out_shape({2, 2, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9 };
    std::vector<float> values_golden = {
        5, 6,
        8, 9 };
    std::vector<uint8_t> indices_golden = {
        3, 2,
        1, 0 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(4);
    std::vector<uint8_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}

TEST(MaxpoolWithArgmax, shape_4_4_1_uint8_kernel_2_stride_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 4, 1});
    tim::vx::ShapeType out_shape({2, 2, 1});
    tim::vx::Quantization io_quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            in_shape, tim::vx::TensorAttribute::INPUT, io_quant);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, io_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<uint8_t> in_data = {
        1, 2, 3, 3,
        4, 5, 6, 6,
        7, 8, 9, 9,
        10, 11, 12, 12 };
    std::vector<uint8_t> values_golden = {
        5, 6,
        11, 12};
    std::vector<uint8_t> indices_golden = {
        3, 2,
        3, 2};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output_values(4);
    std::vector<uint8_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}

