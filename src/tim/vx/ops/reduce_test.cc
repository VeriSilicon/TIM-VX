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
#include "tim/vx/ops.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(Reduce_sum, NotKeepDims) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  float tolerance = ctx->hasSP() ? 0.01568627544 : 1e-5f;

  tim::vx::ShapeType input_shape({2, 3, 1});
  tim::vx::ShapeType output_shape({2, 1});
  tim::vx::Quantization quant_in(tim::vx::QuantType::ASYMMETRIC, 0.00784313772,
                              127);
  tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.01568627544,
                              127);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec dc_spec1(tim::vx::DataType::UINT8, {0, 0, 0},
                               tim::vx::TensorAttribute::TRANSIENT, quant_in);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto dc_tensor1 = graph->CreateTensor(dc_spec1);
  auto dc1_op = graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*dc1_op).BindInputs({input_tensor}).BindOutputs({dc_tensor1});

  tim::vx::TensorSpec reduce_sum_spec(tim::vx::DataType::UINT8, {0, 0, 0},
                                      tim::vx::TensorAttribute::TRANSIENT,
                                      quant_out);
  auto reduce_sum_out = graph->CreateTensor(reduce_sum_spec);
  std::vector<int32_t> axis = {1};
  auto reduce_sum =
      graph->CreateOperation<tim::vx::ops::ReduceSum>(axis, false);
  (*reduce_sum).BindInputs({dc_tensor1}).BindOutputs({reduce_sum_out});

  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto dc2_op = graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*dc2_op).BindInputs({reduce_sum_out}).BindOutputs({output_tensor});

  std::vector<float> in_data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<float> golden = {
      1.2078431,
      1.2078431,
  };

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}

TEST(Reduce_sum, KeepDims) {
  auto ctx = tim::vx::Context::Create();
  if (ctx->isClOnly()) GTEST_SKIP();
  float tolerance = ctx->hasSP() ? 0.01568627544 : 1e-5f;

  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({2, 3});
  tim::vx::ShapeType output_shape({1, 3});
  tim::vx::Quantization quant_in(tim::vx::QuantType::ASYMMETRIC, 0.00784313772,
                              127);
  tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.01568627544,
                              127);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec dc_spec1(tim::vx::DataType::UINT8, {0, 0, 0},
                               tim::vx::TensorAttribute::TRANSIENT, quant_in);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto dc_tensor1 = graph->CreateTensor(dc_spec1);
  auto dc1_op = graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*dc1_op).BindInputs({input_tensor}).BindOutputs({dc_tensor1});

  tim::vx::TensorSpec reduce_sum_spec(tim::vx::DataType::UINT8, {0, 0, 0},
                                      tim::vx::TensorAttribute::TRANSIENT,
                                      quant_out);
  auto reduce_sum_out = graph->CreateTensor(reduce_sum_spec);
  std::vector<int32_t> axis = {0};
  auto reduce_sum = graph->CreateOperation<tim::vx::ops::ReduceSum>(axis, true);
  (*reduce_sum).BindInputs({dc_tensor1}).BindOutputs({reduce_sum_out});

  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto dc2_op = graph->CreateOperation<tim::vx::ops::DataConvert>();
  (*dc2_op).BindInputs({reduce_sum_out}).BindOutputs({output_tensor});

  std::vector<float> in_data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<float> golden = {
      0.596078,
      0.705882,
      1.113725,
  };

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}

TEST(Reduce_all, KeepDims) {
  auto ctx = tim::vx::Context::Create();

  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({2, 3, 2});
  tim::vx::ShapeType output_shape({1, 3, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::BOOL8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<int32_t> axis = {2, 0};
  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto reduce_all = graph->CreateOperation<tim::vx::ops::ReduceAll>(axis, true);
  (*reduce_all).BindInputs({input_tensor}).BindOutputs({output_tensor});

  bool in_data[] = {true, true, true,  true, true, true,
                    true, true, false, true, true, true};
  bool golden[] = {true, false, true};
  bool* p_in = in_data;
  input_tensor->CopyDataToTensor(p_in, 12 * sizeof(bool));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  bool output[3 * sizeof(bool)];
  bool* p_output = output;
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(p_output));
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(golden[i] == output[i]);
  }
}

TEST(Reduce_all, NotKeepDims) {
  auto ctx = tim::vx::Context::Create();

  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({2, 3, 2});
  tim::vx::ShapeType output_shape({2});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::BOOL8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::BOOL8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<int32_t> axis = {1, 2, 2, 2};
  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto reduce_all =
      graph->CreateOperation<tim::vx::ops::ReduceAll>(axis, false);
  (*reduce_all).BindInputs({input_tensor}).BindOutputs({output_tensor});

  bool in_data[] = {true, true, true, true, true, false,
                    true, true, true, true, true, true};
  bool golden[] = {true, false};
  bool* p_in = in_data;
  input_tensor->CopyDataToTensor(p_in, 12 * sizeof(bool));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  bool output[2 * sizeof(bool)];
  bool* p_output = output;
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(p_output));
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(golden[i] == output[i]);
  }
}

TEST(Reduce_max, NotKeepDims) {
  auto ctx = tim::vx::Context::Create();

  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({2, 3});
  tim::vx::ShapeType output_shape({3});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  std::vector<int32_t> axis = {0};
  auto reduce_sum = graph->CreateOperation<tim::vx::ops::ReduceMax>(axis, false);
  (*reduce_sum).BindInputs({input_tensor}).BindOutputs({output_tensor});

  std::vector<float> in_data = {-1.0f, -2.0f, 3.0f, 4.0f, 5.0f, -6.0f};
  std::vector<float> golden = {-1.0f, 4.0f, 5.0f};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}