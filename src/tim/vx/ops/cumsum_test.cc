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
#include "tim/vx/ops/cumsum.h"

#include "gtest/gtest.h"

#ifdef VSI_FEAT_OP_CUMSUM

TEST(CumSum, shape_4_2_fp32_axis_0_exclusive_0_reverse_0) {
  auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({4, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 4, 6, 8,
        1, 3, 5, 7,
        
    };
    std::vector<float> golden = {
       2, 6, 12, 20,
       1, 4, 9, 16,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::CumSum>(0,0,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(CumSum, shape_4_2_fp32_axis_1_exclusive_0_reverse_0) {
  auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({4, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 4, 6, 8,
        1, 3, 5, 7,
        
    };
    std::vector<float> golden = {
       2, 4, 6, 8,
       3, 7, 11,15,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::CumSum>(1,0,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(CumSum, shape_4_1_fp32_axis_0_exclusive_1_reverse_0) {
  auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({4, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 4, 6, 8,

    };
    std::vector<float> golden = {
       0, 2, 6, 12,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::CumSum>(0,1,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(CumSum, shape_4_1_fp32_axis_0_exclusive_1_reverse_1) {
  auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({4, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 4, 6, 8,

    };
    std::vector<float> golden = {
       18, 14, 8, 0,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::CumSum>(0,1,1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(CumSum, shape_4_2_fp32_axis_minus1_exclusive_1_reverse_1) {
  auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({4, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        2, 4, 6, 8,
        1, 3, 5, 7,
        
    };
    std::vector<float> golden = {
       1, 3, 5, 7,
       0, 0, 0, 0,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::CumSum>(-1,1,1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
#endif //(VSI_FEAT_OP_CUMSUM)
