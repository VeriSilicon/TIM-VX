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
#include "tim/vx/ops/arg.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(ArgMax, shape_2_2_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::INT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {2, 1, 3, 10};
  std::vector<int32_t> golden = {0, 1};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::ArgMax>(0);
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(ArgMax, shape_2_2_axis_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::INT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {2, 1, 3, 10};
  std::vector<int32_t> golden = {1, 1};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::ArgMax>(1);
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(ArgMin, shape_2_2_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::INT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {2, 1, 3, 10};
  std::vector<int32_t> golden = {1, 0};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::ArgMin>(0);
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(ArgMin, shape_2_2_axis_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::INT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {2, 1, 3, 10};
  std::vector<int32_t> golden = {0, 0};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::ArgMin>(1);
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}