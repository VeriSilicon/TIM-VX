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
#include "tim/vx/ops/spatial_transformer.h"

#include "gtest/gtest.h"

TEST(SpatialTransformer, shape_1_3_3_1_u8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({1, 3, 3, 1});
    tim::vx::ShapeType theta_shape({6});
    tim::vx::ShapeType out_shape({1, 3, 3, 1});
    tim::vx::Quantization io_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
                            in_shape, tim::vx::TensorAttribute::INPUT, io_quant);
    tim::vx::TensorSpec theta_spec(tim::vx::DataType::UINT8,
                            theta_shape, tim::vx::TensorAttribute::INPUT, io_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, io_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto theta_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {
        2, 4, 6,
        2, 4, 6,
        2, 4, 6 };
    std::vector<uint8_t> theta_data = {
        2, 2, 2,
        2, 2, 2 };
    std::vector<uint8_t> values_golden = {
        2,3,2,
        2,3,2,
        2,3,2 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(theta_tensor->CopyDataToTensor(theta_data.data(), theta_data.size()));
    auto op = graph->CreateOperation<tim::vx::ops::SpatialTransformer>(
        3, 3, true, true, true, true, true, true,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    );
    (*op).BindInputs({input_tensor, theta_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output_values(values_golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
    EXPECT_EQ(values_golden, output_values);
}
