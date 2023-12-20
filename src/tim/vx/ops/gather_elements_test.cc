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
#include "tim/vx/ops/gather_elements.h"
#include <iostream>
#include "gtest/gtest.h"
#include "test_utils.h"


TEST(GatherElements, shape_3_2_1_int32_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2, 1});
  tim::vx::ShapeType indices_shape({2, 2, 1});
  tim::vx::ShapeType out_shape({2, 2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, indices_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int32_t> in_data = {
      1, 2, 3, 4, 5, 6,
  };

  std::vector<int32_t> indices = {
      1,
      2,
      0,
      2,
  };
  std::vector<int32_t> golden = {
      2, 3, 4, 6,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  EXPECT_TRUE(
      indices_tensor->CopyDataToTensor(indices.data(), indices.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::GatherElements>(0);
  (*op).BindInputs({input_tensor, indices_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(GatherElements, shape_3_2_1_int32_axis_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2, 1});
  tim::vx::ShapeType indices_shape({2, 2, 1});
  tim::vx::ShapeType out_shape({2, 2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, indices_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int32_t> in_data = {
      1, 2, 3, 4, 5, 6,
  };

  std::vector<int32_t> indices = {
      1,
      2,
      0,
      2,
  };
  std::vector<int32_t> golden = {
      4, 5, 1, 5,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  EXPECT_TRUE(
      indices_tensor->CopyDataToTensor(indices.data(), indices.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::GatherElements>(1);
  (*op).BindInputs({input_tensor, indices_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(GatherElements, shape_3_2_1_float32_axis_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2, 1});
  tim::vx::ShapeType indices_shape({2, 2, 1});
  tim::vx::ShapeType out_shape({2, 2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec indices_spec(tim::vx::DataType::INT32, indices_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto indices_tensor = graph->CreateTensor(indices_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1, 2, 3, 4, 5, 6,
  };

  std::vector<int32_t> indices = {
      1,
      2,
      0,
      2,
  };
  std::vector<float> golden = {
      1, 2, 3, 4,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  EXPECT_TRUE(
      indices_tensor->CopyDataToTensor(indices.data(), indices.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::GatherElements>(2);
  (*op).BindInputs({input_tensor, indices_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

