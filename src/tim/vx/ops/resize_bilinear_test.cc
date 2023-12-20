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
#include "tim/vx/ops/resize.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(ResizeBilinear, align_corners) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({3, 3, 1, 1});
    tim::vx::ShapeType output_shape({2, 2, 1, 1});

    std::vector<float> scales_input = {0.5f};
    std::vector<int32_t> zero_point_input = {128};
    std::vector<float> scales_output = {0.5f};
    std::vector<int32_t> zero_point_output = {128};

    tim::vx::Quantization quant_in(tim::vx::QuantType::ASYMMETRIC, 0,
                                   scales_input, zero_point_input);
    tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0,
                                    scales_output, zero_point_output);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                   tim::vx::TensorAttribute::INPUT, quant_in);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                    tim::vx::TensorAttribute::OUTPUT, quant_out);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {130, 132, 134, 136, 138,
                                    140, 142, 144, 146};
    std::vector<uint8_t> golden = {130, 134, 142, 146};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

    auto op = graph->CreateOperation<tim::vx::ops::Resize>(
        tim::vx::ResizeType::BILINEAR, 0, true, false, 2, 2);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
