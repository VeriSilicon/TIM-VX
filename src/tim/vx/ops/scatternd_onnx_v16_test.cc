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
#include "tim/vx/ops/scatternd_onnx_v16.h"

#include "gtest/gtest.h"

TEST(ScatterND_ONNX_V16, shape_8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({8});
    tim::vx::ShapeType indices_shape({1,4});
    tim::vx::ShapeType updates_shape({4});
    tim::vx::ShapeType out_shape({8});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32,
                            indices_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            updates_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto indices_tensor = graph->CreateTensor(indices_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> input_data = { 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> indices_data = { 4, 3, 1, 7 };
    std::vector<float> updates_data = { 9, 10, 11, 12 };
    std::vector<float> golden = { 1, 11, 3, 10, 9, 6, 7, 12 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        input_data.data(), input_data.size()*sizeof(float)));
    EXPECT_TRUE(indices_tensor->CopyDataToTensor(
        indices_data.data(), indices_data.size()*sizeof(int32_t)));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size()*sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::ScatterND_ONNX_V16>();
    (*op).BindInputs({input_tensor, indices_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

