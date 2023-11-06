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
#include "tim/vx/ops/gather.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(Gather, shape_5_3_2_2_int32_axis_1_batchdims_1) {
  auto ctx = tim::vx::Context::Create();

  if (ctx->isClOnly()) GTEST_SKIP();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({5, 3, 2, 2});
  tim::vx::ShapeType indices_shape({2, 2, 2});
  tim::vx::ShapeType out_shape({5, 2, 2, 2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, indices_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int8_t> in_data = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
      45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
  };

  std::vector<int32_t> indices = {1, 0, 0, 1, 1, 0, 0, 1};
  std::vector<int8_t> golden = {
      5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 30, 31, 32,
      33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53,
      54, 45, 46, 47, 48, 49, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(indices_tensor->CopyDataToTensor(indices.data(), indices.size()));
  auto op = graph->CreateOperation<tim::vx::ops::Gather>(1, 1);
  (*op).BindInputs({input_tensor, indices_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int8_t> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Gather, shape_2_2_indices_2) {
  auto ctx = tim::vx::Context::Create();

  if (ctx->isClOnly()) GTEST_SKIP();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType indices_shape({2});
  tim::vx::ShapeType out_shape({2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, indices_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-2.0f, 0.2f, 0.7f, 0.8f};

  std::vector<int32_t> indices = {1, 0};
  std::vector<float> golden = {0.7f, 0.8f, -2.0f, 0.2f};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(indices_tensor->CopyDataToTensor(indices.data(), indices.size()));
  auto op = graph->CreateOperation<tim::vx::ops::Gather>(1, 0);
  (*op).BindInputs({input_tensor, indices_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Gather, scalar_index_input2D) {
  auto ctx = tim::vx::Context::Create();

  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({5,2});
  tim::vx::ShapeType index_shape({1});
  tim::vx::ShapeType out_shape({5});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec index_spec(tim::vx::DataType::INT32, index_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto index_tensor = graph->CreateTensor(index_spec);
  index_tensor->SetScalar(1);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
    1,2,3,4,5,
    6,7,8,9,10};

  std::vector<int32_t> index = {1};
  std::vector<float> golden = {6,7,8,9,10};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(index_tensor->CopyDataToTensor(index.data(), index.size()));
  auto op = graph->CreateOperation<tim::vx::ops::Gather>(1, 0);
  (*op).BindInputs({input_tensor, index_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Gather, scalar_index_input1D) {
  auto ctx = tim::vx::Context::Create();

  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({5});
  tim::vx::ShapeType index_shape({1});
  tim::vx::ShapeType out_shape({});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec index_spec(tim::vx::DataType::INT32, index_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec gatherout_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                   tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto index_tensor = graph->CreateTensor(index_spec);
  index_tensor->SetScalar(1);
  auto gatherout_tensor = graph->CreateTensor(gatherout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
    1,2,3,4,5};

  std::vector<int32_t> index = {1};
  std::vector<float> golden = {2};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(index_tensor->CopyDataToTensor(index.data(), index.size()));
  auto gather = graph->CreateOperation<tim::vx::ops::Gather>(0);
  (*gather).BindInputs({input_tensor, index_tensor}).BindOutputs({gatherout_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(gatherout_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}