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
#include "tim/vx/ops/l2normalization.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(L2Norm, axis_1_shape_2_3_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({2,3});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {0.0f, 3.0f, 3.0f, 0.0f, 4.0f, 4.0f};
    std::vector<float> golden = {0.0f, 0.6f, 0.6f, 0.0f, 0.8f, 0.8f};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::L2Normalization>(1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(6);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}