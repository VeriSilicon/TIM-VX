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
#include "tim/vx/ops/topk.h"

#include "gtest/gtest.h"

TEST(Topk, shape_3_4_k_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3});
  tim::vx::ShapeType values_shape({6});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec values_spec(tim::vx::DataType::FLOAT32, values_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, values_shape,
                                   tim::vx::TensorAttribute::OUTPUT);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto value_tensor = graph->CreateTensor(values_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);

  std::vector<float> in_data = {
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
  };
  std::vector<float> value_golden = {
      4, 3, 4, 3, 4, 3,
  };
  std::vector<int32_t> indices_golden = {
      3, 2, 3, 2, 3, 2,
  };
  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Topk>(2);
  (*op).BindInputs({input_tensor}).BindOutputs({value_tensor, indices_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> value(value_golden.size());
  std::vector<int32_t> indices(6);
  EXPECT_TRUE(value_tensor->CopyDataFromTensor(value.data()));
  EXPECT_TRUE(indices_tensor->CopyDataFromTensor(indices.data()));
  EXPECT_EQ(value_golden, value);
  EXPECT_EQ(indices_golden, indices);
}

TEST(Topk, shape_3_2_2_k_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 3});
  tim::vx::ShapeType values_shape({6});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec values_spec(tim::vx::DataType::FLOAT32, values_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, values_shape,
                                   tim::vx::TensorAttribute::OUTPUT);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto value_tensor = graph->CreateTensor(values_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);

  std::vector<float> in_data = {
      1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
  };
  std::vector<float> value_golden = {
      2, 4, 2, 4, 2, 4,
  };
  std::vector<int32_t> indices_golden = {
      1, 1, 1, 1, 1, 1,
  };
  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Topk>(1);
  (*op).BindInputs({input_tensor}).BindOutputs({value_tensor, indices_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> value(value_golden.size());
  std::vector<int32_t> indices(6);
  EXPECT_TRUE(value_tensor->CopyDataFromTensor(value.data()));
  EXPECT_TRUE(indices_tensor->CopyDataFromTensor(indices.data()));
  EXPECT_EQ(value_golden, value);
  EXPECT_EQ(indices_golden, indices);
}