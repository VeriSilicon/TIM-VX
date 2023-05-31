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
#include "tim/vx/ops/pad.h"
#include "tim/vx/ops/pad_v2.h"
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
      0, 1, 2, 3, 4, 5,
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

TEST(Pad, float_1_3_2_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 1});
  tim::vx::ShapeType output_shape({1, 7, 4, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
  };

  std::vector<float> golden = {
      9.3f, 1.0f, 2.0f, 3.0f, 9.3f, 9.3f, 9.3f, 9.3f, 4.0f, 5.0f, 6.0f, 9.3f, 9.3f, 9.3f,
      9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f, 9.3f
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {0, 1, 0, 0};
  std::vector<uint32_t> back = {0, 3, 2, 0};
  auto op = graph->CreateOperation<tim::vx::ops::PadV2>(
      front, back, 9.3f, tim::vx::ops::PadV2::PAD_MODE_CONSTANT);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Pad, int8_1_3_2_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 1});
  tim::vx::ShapeType output_shape({1, 7, 4, 1});
  float scales = 2.3f;
  int zero_point = -124;

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC,
                                    scales, zero_point);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC,
                                     scales, zero_point);
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT, quant_output);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int8_t> input_data = {
      -127, -126, -125, -124, -123, -122
  };

  std::vector<int8_t> golden = {
      -120, -127, -126, -125, -120, -120, -120, -120, -124, -123, -122, -120, -120, -120,
      -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120, -120,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {0, 1, 0, 0};
  std::vector<uint32_t> back = {0, 3, 2, 0};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, 9, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<int8_t> output(golden.size());

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
      0, 1, 2, 3, 4, 5,
  };

  std::vector<float> golden = {
      4, 3, 4, 5, 4, 1, 0, 1, 2, 1, 4, 3, 4, 5, 4, 1, 0, 1, 2, 1
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  std::vector<uint32_t> front = {1, 1};
  std::vector<uint32_t> back = {1, 1};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, 0, tim::vx::ops::Pad::PAD_MODE_REFLECT);
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
      0, 1, 2, 3, 4, 5,
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