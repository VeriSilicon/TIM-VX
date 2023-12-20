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
#include "tim/vx/ops/fullyconnected.h"
#include <iostream>
#include "gtest/gtest.h"
#include "test_utils.h"
#include "third_party/half/half.hpp"

TEST(FullyConnected, unit_2_float_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType weight_shape({2, 3});
  tim::vx::ShapeType bias_shape({3});
  tim::vx::ShapeType out_shape({3, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  std::vector<float> in_data = {
      1,4,2,6,
  };
  std::vector<float> weight = {
      -3,3,2,1,0,4,
  };
  std::vector<float> bias = {
      0.1, 0.4, 0.6,
  };
  std::vector<float> golden = {
      9.1, 6.4, 16.6, 12.1, 10.4, 24.6,
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::FullyConnected>(0, 3);
  (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(FullyConnected, unit_2_float16_axis_0) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  using namespace half_float::literal;

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType weight_shape({2, 3});
  tim::vx::ShapeType bias_shape({3});
  tim::vx::ShapeType out_shape({3, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT16, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT16, weight_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT16, bias_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT16, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  std::vector<half_float::half> in_data = {
      1.0_h, 4.0_h, 2.0_h, 6.0_h
  };
  std::vector<half_float::half> weight = {
      -3.0_h, 3.0_h, 2.0_h, 1.0_h, 0.0_h, 4.0_h
  };
  std::vector<half_float::half> bias = {
      0.1_h, 0.4_h, 0.6_h
  };
  std::vector<half_float::half> golden = {
      9.1_h, 6.4_h, 16.6_h, 12.1_h, 10.4_h, 24.6_h
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data()));
  auto op = graph->CreateOperation<tim::vx::ops::FullyConnected>(0, 3);
  (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<half_float::half> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, (half_float::half)0.1));
}