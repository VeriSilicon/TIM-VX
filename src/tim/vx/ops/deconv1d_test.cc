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
#include "tim/vx/ops/deconv1d.h"
#include "tim/vx/ops/activations.h"

#include "gtest/gtest.h"

TEST(DeConv1d, no_bias_no_outputpadding_shape_3_2_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 1});
  tim::vx::ShapeType kernel_shape({3, 2, 1});
  tim::vx::ShapeType output_shape({5, 1, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto kernel_tensor = graph->CreateTensor(kernel_spec);

  std::vector<float> input_data = {
      3.0f, 9.0f, 3.0f, 7.0f, 5.0f, 9.0f,
  };
  std::vector<float> kernel_data = {
      9.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f,
  };

  std::vector<float> golden = {48, 96, 57, 9, 3};

  std::vector<float> output_data(golden.size());

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(),
                                              kernel_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
      2, tim::vx::PadType::VALID, /*ksize=*/3, /*stride=*/1,
      /*output_padding=*/0, std::array<uint32_t, 2>({0, 0}), 1);
  (*op).BindInputs({input_tensor, kernel_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));

  EXPECT_EQ(golden, output_data) << "Result mismatch";
}

TEST(DeConv1d, no_bias_has_outputpadding_shape_3_2_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 1});
  tim::vx::ShapeType kernel_shape({3, 2, 1});
  tim::vx::ShapeType output_shape({6, 1, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto kernel_tensor = graph->CreateTensor(kernel_spec);

  std::vector<float> input_data = {
      3.0f, 9.0f, 3.0f, 7.0f, 5.0f, 9.0f,
  };
  std::vector<float> kernel_data = {
      9.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f,
  };

  std::vector<float> golden = {48, 96, 57, 9, 3, 0};

  std::vector<float> output_data(golden.size());

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(),
                                              kernel_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
      2, tim::vx::PadType::VALID, /*ksize=*/3, /*stride=*/1,
      /*output_padding=*/ 1, std::array<uint32_t, 2>({0, 0}), 1);
  (*op).BindInputs({input_tensor, kernel_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));

  EXPECT_EQ(golden, output_data) << "Result mismatch";
}

TEST(DeConv1d, layout_wcn_shape_3_1_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 1, 1});
  tim::vx::ShapeType kernel_shape({3, 1, 1});
  tim::vx::ShapeType output_shape({5, 1, 1});
  tim::vx::ShapeType bias_shape({1});

  tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
  tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 2);
  tim::vx::Quantization weight_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);
  tim::vx::Quantization bias_quant(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, input_quant);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::UINT8, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  weight_quant);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, bias_quant);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  output_quant);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto kernel_tensor = graph->CreateTensor(kernel_spec);
  auto bias_tensor = graph->CreateTensor(bias_spec);

  std::vector<uint8_t> input_data = {
      3,
      9,
      3,
  };

  std::vector<uint8_t> kernel_data = {
      9,
      0,
      1,
  };

  std::vector<int32_t> bias_data = {
      -5,
  };

  std::vector<uint8_t> golden = {
      24, 78, 27, 6, 0,
  };

  std::vector<uint8_t> output_data(golden.size());

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size()));
  EXPECT_TRUE(
      kernel_tensor->CopyDataToTensor(kernel_data.data(), kernel_data.size()));
  EXPECT_TRUE(bias_tensor->CopyDataToTensor(
      bias_data.data(), bias_data.size() * sizeof(int32_t)));

  auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
      1, tim::vx::PadType::VALID, /*ksize=*/3, /*stride=*/1,
      /*output_padding=*/0,
      std::array<uint32_t, 2>({
          0,
          0,
      }),
      /*group=*/1);
  (*op)
      .BindInputs({input_tensor, kernel_tensor, bias_tensor})
      .BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));
  EXPECT_EQ(golden, output_data) << "Result mismatch";
}

TEST(DeConv1dLeakyRelu, shape_5_5_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({5, 5, 1});  //wcn
  tim::vx::ShapeType kernel_shape({3, 5, 2});
  tim::vx::ShapeType output_shape({7, 2, 1});  //wcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec transisent_spec(tim::vx::DataType::FLOAT32, {},
                                      tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto transient_tensor = graph->CreateTensor(transisent_spec);
  auto kernel_tensor = graph->CreateTensor(kernel_spec);

  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,
                                   8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0,
                                   15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
                                   22.0, 23.0, 24.0, 25.0};
  std::vector<float> kernel_data = {1,  2,  3,  7,  8,  9,  13, 14, 15, 19,
                                    20, 21, 25, 26, 27, 4,  5,  6,  10, 11,
                                    12, 16, 17, 18, 22, 23, 24, 28, 29, 30};
  std::vector<float> golden = {
      1015.0, 2150.0, 3410.0, 3620.0, 3830.0, 2700.0, 1425.0,
      1180.0, 2495.0, 3950.0, 4205.0, 4460.0, 3135.0, 1650.0,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(),
                                              kernel_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::DeConv1d>(
      1, tim::vx::PadType::VALID, /*ksize=*/3, /*stride=*/1,
      /*output_padding=*/0, std::array<uint32_t, 2>({0, 0}), /*group=*/1);
  (*op)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({transient_tensor});

  auto leakyrelu = graph->CreateOperation<tim::vx::ops::LeakyRelu>(0.01f);
  (*leakyrelu).BindInputs({transient_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output_data(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));

  EXPECT_EQ(golden, output_data) << "Result mismatch";
}