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
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/relational_operations.h"

#include "gtest/gtest.h"

TEST(Reshape, 3_reshape) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType a_shape({1, 2});
  tim::vx::ShapeType out_shape({1, 2});

  tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32, a_shape,
                             tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec a_spec0(tim::vx::DataType::FLOAT32, a_shape,
                              tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto a_tensor = graph->CreateTensor(a_spec);
  auto a_tensor0 = graph->CreateTensor(a_spec0);
  auto a_tensor1 = graph->CreateTensor(a_spec0);
  auto out_tensor = graph->CreateTensor(out_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op).BindInputs({a_tensor}).BindOutputs({a_tensor0});

  auto op1 = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op1).BindInputs({a_tensor0}).BindOutputs({a_tensor1});

  auto op2 = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op2).BindInputs({a_tensor1}).BindOutputs({out_tensor});

  uint32_t data_size = 2;
  std::vector<float> input_a_data{3, -1};
  a_tensor->CopyDataToTensor(
        input_a_data.data(), input_a_data.size() * 4);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> golden{3, -1};
  std::vector<float> output(data_size, 0);
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Reshape, reshape_for_equal_i32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType a_shape({16});
  tim::vx::ShapeType out_shape({16});

  tim::vx::TensorSpec a_spec(tim::vx::DataType::INT32, a_shape,
                             tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec a_spec0(tim::vx::DataType::INT32, a_shape,
                              tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec0(tim::vx::DataType::BOOL8, out_shape,
                              tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::BOOL8, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto a_tensor = graph->CreateTensor(a_spec);
  auto b_tensor = graph->CreateTensor(a_spec);
  auto a_tensor0 = graph->CreateTensor(a_spec0);
  auto b_tensor0 = graph->CreateTensor(a_spec0);
  auto out_tensor0 = graph->CreateTensor(out_spec0);
  auto out_tensor = graph->CreateTensor(out_spec);

  auto op = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op).BindInputs({a_tensor}).BindOutputs({a_tensor0});

  auto op1 = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op1).BindInputs({b_tensor}).BindOutputs({b_tensor0});

  auto op2 = graph->CreateOperation<tim::vx::ops::Equal>();
  (*op2).BindInputs({a_tensor0, b_tensor0}).BindOutputs({out_tensor0});

  auto op3 = graph->CreateOperation<tim::vx::ops::Reshape>(a_shape);
  (*op3).BindInputs({out_tensor0}).BindOutputs({out_tensor});

  uint32_t data_size = 16;
  std::vector<int32_t> input_a_data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
  std::vector<int32_t> input_b_data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
  a_tensor->CopyDataToTensor(
        input_a_data.data(), input_a_data.size() * 4);
  b_tensor->CopyDataToTensor(
        input_b_data.data(), input_b_data.size() * 4);

  graph->PrintGraph();

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  //Not using vector<bool> because it uses a bitfield representation internally
  //and it's cumbersome to copy tensor data to it.
  std::vector<uint8_t> golden(data_size, 1);
  std::vector<uint8_t> output(data_size, 0);
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
