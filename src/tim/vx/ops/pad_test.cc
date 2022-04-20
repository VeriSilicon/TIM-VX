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
#include "tim/vx/ops/pad.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(Pad, constant) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2});
  tim::vx::ShapeType output_shape({5, 4});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {
      0, 1, 2, 3, 4, 5, 6,
  };

  std::vector<float> golden = {
      1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 3, 4, 5, 1, 1, 1, 1, 1, 1,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {1, 1};
  std::vector<uint32_t> back = {1, 1};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, 1, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Pad, reflect) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2});
  tim::vx::ShapeType output_shape({5, 4});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {
      0, 1, 2, 3, 4, 5, 6,
  };

  std::vector<float> golden = {
      0, 0, 1, 2, 2, 0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 3, 3, 4, 5, 5,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {1, 1};
  std::vector<uint32_t> back = {1, 1};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, 0, tim::vx::ops::Pad::PAD_MODE_EDGE);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Pad, edge) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2});
  tim::vx::ShapeType output_shape({5, 4});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {
      0, 1, 2, 3, 4, 5, 6,
  };

  std::vector<float> golden = {0, 0, 1, 2, 2, 0, 0, 1, 2, 2,
                               3, 3, 4, 5, 5, 3, 3, 4, 5, 5};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {1, 1};
  std::vector<uint32_t> back = {1, 1};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, 0, tim::vx::ops::Pad::PAD_MODE_EDGE);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}