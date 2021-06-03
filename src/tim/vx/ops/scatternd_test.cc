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
#include "tim/vx/ops/scatternd.h"

#include "gtest/gtest.h"

TEST(ScatterND, shape_4_4_4) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType indices_shape({1,2});
    tim::vx::ShapeType updates_shape({4,4,2});
    tim::vx::ShapeType out_shape({4, 4, 4});
    tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32,
                            indices_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::FLOAT32,
                            updates_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto indices_tensor = graph->CreateTensor(indices_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> indices_data = { 0, 2 };
    std::vector<float> updates_data = {
        5,5,5,5, 6,6,6,6,
        7,7,7,7, 8,8,8,8,
        1,1,1,1, 2,2,2,2,
        3,3,3,3, 4,4,4,4,
        };
    std::vector<float> golden = {
        5,5,5,5, 6,6,6,6,
        7,7,7,7, 8,8,8,8,
        0,0,0,0, 0,0,0,0,
        0,0,0,0, 0,0,0,0,
        1,1,1,1, 2,2,2,2,
        3,3,3,3, 4,4,4,4,
        0,0,0,0, 0,0,0,0,
        0,0,0,0, 0,0,0,0,
        };

    EXPECT_TRUE(indices_tensor->CopyDataToTensor(
        indices_data.data(), indices_data.size()*sizeof(int32_t)));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size()*sizeof(int32_t)));
    std::vector<uint32_t> shape = {4, 4, 4};
    auto op = graph->CreateOperation<tim::vx::ops::ScatterND>(shape);
    (*op).BindInputs({indices_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(ScatterND, shape_9) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType indices_shape({4});
    tim::vx::ShapeType updates_shape({4});
    tim::vx::ShapeType out_shape({9});
    tim::vx::Quantization updates_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32,
                            indices_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec updates_spec(tim::vx::DataType::UINT8,
                            updates_shape, tim::vx::TensorAttribute::INPUT, updates_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);

    auto indices_tensor = graph->CreateTensor(indices_spec);
    auto updates_tensor = graph->CreateTensor(updates_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> indices_data = { 4, 3, 1, 7 };
    std::vector<uint8_t> updates_data = {
        18, 20, 22, 24
        };
    std::vector<uint8_t> golden = {
        0, 22, 0, 20, 18, 0, 0, 24, 0 
        };

    EXPECT_TRUE(indices_tensor->CopyDataToTensor(
        indices_data.data(), indices_data.size()));
    EXPECT_TRUE(updates_tensor->CopyDataToTensor(
        updates_data.data(), updates_data.size()));
    std::vector<uint32_t> shape = {9};
    auto op = graph->CreateOperation<tim::vx::ops::ScatterND>(shape);
    (*op).BindInputs({indices_tensor, updates_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
