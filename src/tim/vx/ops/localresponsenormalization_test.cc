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
#include "tim/vx/ops/localresponsenormalization.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(localresponsenormalization, axis_0_shape_6_1_1_1_float) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({6, 1, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
  std::vector<float> golden = {-0.264926, 0.125109,  0.140112,
                               0.267261,  -0.161788, 0.0244266};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  int radius = 2;
  int size = radius * 2 + 1;
  float alpha = 4.0, beta = 0.5, bias = 9.0;
  auto op = graph->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
      size, alpha, beta, bias, 0);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(localresponsenormalization, axis_1_shape_2_6_float) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({2, 6});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-1.100000023841858f, -1.100000023841858f, 0.6000000238418579f,
                                0.6000000238418579f, 0.699999988079071f, 0.699999988079071f,
                                1.2000000476837158f, 1.2000000476837158f, -0.699999988079071f,
                                -0.699999988079071f, 0.10000000149011612f, 0.10000000149011612f};
  std::vector<float> golden = {-0.26492568850517273f, -0.26492568850517273f, 0.12510864436626434f,
                                0.12510864436626434f, 0.14011213183403015f, 0.14011213183403015f,
                                0.267261266708374f, 0.267261266708374f, -0.16178755462169647f,
                                -0.16178755462169647f, 0.024426599964499474f, 0.024426599964499474f};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  int radius = 2;
  int size = radius * 2 + 1;
  float alpha = 4.0, beta = 0.5, bias = 9.0;
  auto op = graph->CreateOperation<tim::vx::ops::LocalResponseNormalization>(
      size, alpha, beta, bias, 1);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}