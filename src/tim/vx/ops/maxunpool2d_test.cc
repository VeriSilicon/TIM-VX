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
#include "tim/vx/ops/maxunpool2d.h"

#include "gtest/gtest.h"

TEST(MaxUnpool2d, shape_2_2_1_fp32_kernel_2_stride_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({2, 2, 1});
    tim::vx::ShapeType out_shape({3, 3, 1});
    tim::vx::TensorSpec values_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec indices_spec(tim::vx::DataType::UINT8,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto values_tensor = graph->CreateTensor(values_spec);
    auto indices_tensor = graph->CreateTensor(indices_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> values = {
        5, 6,
        8, 9 };
    std::vector<uint8_t> indices = {
        3, 2,
        1, 0 };
    std::vector<float> golden = {
        0, 0, 0,
        0, 5, 6,
        0, 8, 9 };

    EXPECT_TRUE(values_tensor->CopyDataToTensor(values.data(), values.size()*4));
    EXPECT_TRUE(indices_tensor->CopyDataToTensor(indices.data(), indices.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxUnpool2d>(ksize, stride);
    (*op).BindInputs({values_tensor, indices_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(MaxUnpool2d, shape_2_2_1_uint8_kernel_2_stride_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({2, 2, 1});
    tim::vx::ShapeType out_shape({4, 4, 1});
    tim::vx::Quantization io_quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::TensorSpec values_spec(tim::vx::DataType::UINT8,
                            in_shape, tim::vx::TensorAttribute::INPUT, io_quant);
    tim::vx::TensorSpec indices_spec(tim::vx::DataType::UINT8,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, io_quant);

    auto values_tensor = graph->CreateTensor(values_spec);
    auto indices_tensor = graph->CreateTensor(indices_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> values = {
        5, 6,
        11, 12};
    std::vector<uint8_t> indices = {
        3, 2,
        3, 2};
    std::vector<uint8_t> golden = {
        0, 0, 0, 0,
        0, 5, 6, 0,
        0, 0, 0, 0,
        0, 11, 12, 0 };

    EXPECT_TRUE(values_tensor->CopyDataToTensor(values.data(), values.size()));
    EXPECT_TRUE(indices_tensor->CopyDataToTensor(indices.data(), indices.size()));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxUnpool2d>(ksize, stride);
    (*op).BindInputs({values_tensor, indices_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
