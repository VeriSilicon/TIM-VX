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
#include "tim/vx/ops/conv2d.h"

#include "gtest/gtest.h"
#include "test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"
#include "third_party/half/half.hpp"

TEST(Conv2d, shape_4_2_1_1_float16_PaddingTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  using namespace half_float::literal;

  tim::vx::ShapeType input_shape({4, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT16, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT16, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT16, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT16, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw

  std::vector<half_float::half> input_data = {
      1.0_h, 1.0_h, 1.0_h, 1.0_h,  // row = 1
      2.0_h, 2.0_h, 3.0_h, 2.0_h   // row = 2
  };

  // weight data   oihw
  std::vector<half_float::half> weight_data = {
      1.0_h,  2.0_h,  3.0_h,  4.0_h,  //first 2x2 filter
      -1.0_h, 1.0_h,  -1.0_h, 1.0_h,  // second 2x2 filter
      -1.0_h, -1.0_h, 1.0_h,  1.0_h,  // third 2x2 filter
  };

  // bias data
  std::vector<half_float::half> bias_data = {1.0_h, 2.0_h, 3.0_h};

  std::vector<half_float::half> golden = {
      // first channel
      18.0_h, 22.0_h, 21.0_h, 8.0_h, 7.0_h, 9.0_h, 8.0_h, 3.0_h, 2.0_h, 3.0_h,
      1.0_h, -1.0_h,
      // second channel
      2.0_h, 3.0_h, 1.0_h, 0.0_h, 5.0_h, 6.0_h, 6.0_h, 4.0_h, -1.0_h, -2.0_h,
      -2.0_h, 1.0_h};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<half_float::half> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, (half_float::half)0.1));
}

TEST(Conv2d, shape_4_2_1_1_float32_PaddingTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      1, 1, 1, 1,  // row = 1
      2, 2, 3, 2   // row = 2
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1,  2,  3,  4,  //first 2x2 filter
      -1, 1,  -1, 1,  // second 2x2 filter
      -1, -1, 1,  1,  // third 2x2 filter
  };

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {// first channel
                               18, 22, 21, 8, 7, 9, 8, 3, 2, 3, 1, -1,
                               // second channel
                               2, 3, 1, 0, 5, 6, 6, 4, -1, -2, -2, 1};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_2_2_float32_PointwiseTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({1, 1, 2, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2  // first filter
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3,
                               1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_float32_SimpleTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  };

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_2_2_float32_SimpleChannelsTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data
  std::vector<float> input_data = {
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data
  std::vector<float> weight_data = {1,  2, 3,  4, 1,  2,  3, 4, -1, 1,  -1, 1,
                                    -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1,  1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  std::vector<float> golden = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_6_3_1_1_float32_SimpleAnisotropicStridesTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({6, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {3,  2,  1,  -1, -2, -3, 4,  3,  2,
                                   -2, -3, -4, 5,  4,  3,  -3, -4, -5};

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2,  //
      3, 4,  //
  };

  // bias data
  std::vector<float> bias_data = {-1};

  // nchw
  std::vector<float> golden = {
      30, -24,  //
      40, -34,  //
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({3, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {105, 150, 183, 95,  235, 312,
                               357, 178, 187, 234, 261, 121};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedConstFilterTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {105, 150, 183, 95,  235, 312,
                               357, 178, 187, 234, 261, 121};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedBiasTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {10};

  // nchw
  std::vector<float> golden = {115, 160, 193, 105, 245, 322,
                               367, 188, 197, 244, 271, 131};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {312, 357};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, DISABLED_shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({1, 1, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 2, 3};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3, 2.5, 2.5, 2.5, 2.5, 5,   5, 5,   5,
      1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6, 2.5, 5,   7.5, 10,  2.5, 5, 7.5, 10};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_9_9_1_1_float32_SimpleDilationTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({3, 3});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_float32_StrideTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 2, 2, 3, 2,
                                   1, 2, 3, 4, 1, 2, 4, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {18, 22, 21, 2, 3, 1, 5, 6, 6,
                               17, 31, 40, 4, 5, 3, 3, 4, 4};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({4, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 2, 2, 2, 2,
                                   1, 2, 3, 4, 1, 2, 3, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {10, 34};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_uint8_QuantizedTest1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -63.5, input_max = 64, weight_min = -63.5, weight_max = 64,
        output_min = -127, output_max = 128;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<u_int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<u_int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {scales_zp.first};
  std::vector<int32_t> zero_point_weight = {scales_zp.second};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0]};
  std::vector<int32_t> zero_point_bias = {0};

  scales_zp = QuantizationParams<u_int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {1, 1, 1, 1, 2, 2, 2, 2,
                                         1, 2, 3, 4, 1, 2, 3, 4};
  // weight data   oihw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> weight_data_float = {1,  2, 3,  4,  -1, 1,
                                          -1, 1, -1, -1, 1,  1};
  // bias data
  // scale:0.25  Zp:0
  std::vector<float> bias_data_float = {1, 2, 3};
  // golden data
  //min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scales_input[0], zero_point_input[0]);
  std::vector<u_int8_t> weight_data = Quantize<uint8_t>(
      weight_data_float, scales_weight[0], zero_point_input[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scales_bias[0], zero_point_bias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scales_output[0], zero_point_output[0]);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_uint8_QuantizedTest2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -128.5, input_max = 128, weight_min = -128.5,
        weight_max = 128, output_min = -127, output_max = 128;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<u_int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<u_int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {scales_zp.first};
  std::vector<int32_t> zero_point_weight = {scales_zp.second};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0]};
  std::vector<int32_t> zero_point_bias = {0};

  scales_zp = QuantizationParams<u_int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-128.5  max:128  scale:1.00588  Zp:0
  std::vector<float> input_data_float = {1, 1, 1, 1, 2, 2, 2, 2,
                                         1, 2, 3, 4, 1, 2, 3, 4};
  // weight data   oihw
  // min:-128.5  max:128  scale:1.00588  Zp:0
  std::vector<float> weight_data_float = {1,  2, 3,  4,  -1, 1,
                                          -1, 1, -1, -1, 1,  1};
  // bias data
  // scale:1.0116  Zp:0
  std::vector<float> bias_data_float = {1, 2, 3};
  // golden data
  // min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scales_input[0], zero_point_input[0]);
  std::vector<u_int8_t> weight_data = Quantize<uint8_t>(
      weight_data_float, scales_weight[0], zero_point_input[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scales_bias[0], zero_point_bias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scales_output[0], zero_point_output[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({6, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 2, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -63.5, input_max = 64, weight_min = -63.5, weight_max = 64,
        output_min = -127, output_max = 128;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<u_int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<u_int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {scales_zp.first};
  std::vector<int32_t> zero_point_weight = {scales_zp.second};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0]};
  std::vector<int32_t> zero_point_bias = {0};

  scales_zp = QuantizationParams<u_int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3,  2,  1,  -1, -2, -3, 4,  3,  2,
                                         -2, -3, -4, 5,  4,  3,  -3, -4, -5};
  // weight data   oihw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> weight_data_float = {1, 2, 3, 4};
  // bias data
  // scale:0.25  Zp:0
  std::vector<float> bias_data_float = {-1};
  // golden data
  //min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {30, -24, 40, -34};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scales_input[0], zero_point_input[0]);
  std::vector<u_int8_t> weight_data = Quantize<uint8_t>(
      weight_data_float, scales_weight[0], zero_point_input[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scales_bias[0], zero_point_bias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scales_output[0], zero_point_output[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({3, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_9_9_1_1_uint8_DilationQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -128, input_max = 127, weight_min = -128, weight_max = 127,
        output_min = 0, output_max = 255;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<u_int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<u_int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {scales_zp.first};
  std::vector<int32_t> zero_point_weight = {scales_zp.second};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0]};
  std::vector<int32_t> zero_point_bias = {0};

  scales_zp = QuantizationParams<u_int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-128  max:127  scale:1  Zp:0
  std::vector<float> input_data_float = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // weight data   oihw
  // min:-128  max:127  scale:1  Zp:0
  std::vector<float> weight_data_float = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // bias data
  // scale:1  Zp:0
  std::vector<float> bias_data_float = {0};
  // golden data
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> golden_float = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scales_input[0], zero_point_input[0]);
  std::vector<u_int8_t> weight_data = Quantize<uint8_t>(
      weight_data_float, scales_weight[0], zero_point_input[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scales_bias[0], zero_point_bias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scales_output[0], zero_point_output[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({3, 3});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_3_2_2_1_int8_QuantizedPerTensorTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -63.5, input_max = 64, weight_min = -63.5, weight_max = 64,
        output_min = -63.5, output_max = 64;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {1};
  std::vector<int32_t> zero_point_weight = {0};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0]};
  std::vector<int32_t> zero_point_bias = {0};

  scales_zp = QuantizationParams<int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scales_input[0], zero_point_input[0]);

  // weight_float_data = {1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2};
  std::vector<int8_t> weight_data = {1, 3, 3, 5, 2, 4, 4, 6,
                                     7, 5, 3, 1, 8, 6, 4, 2};

  // bias data
  std::vector<float> bias_data_float = {3, -2};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scales_bias[0], zero_point_bias[0]);

  // golden_int8_data = {61, -115, 111, -89}
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> golden_float = {31, -57, 56, -44};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scales_output[0], zero_point_output[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_3_2_2_1_int8_QuantizedPerChannelTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float input_min = -63.5, input_max = 64, weight_min = 0, weight_max = 0,
        output_min = -63.5, output_max = 64;

  std::pair<float, int32_t> scales_zp;

  scales_zp = QuantizationParams<int8_t>(input_min, input_max);
  std::vector<float> scales_input = {scales_zp.first};
  std::vector<int32_t> zero_point_input = {scales_zp.second};

  scales_zp = QuantizationParams<int8_t>(weight_min, weight_max);
  std::vector<float> scales_weight = {1, 2};
  std::vector<int32_t> zero_point_weight = {0, 0};

  std::vector<float> scales_bias = {scales_input[0] * scales_weight[0],
                                    scales_input[0] * scales_weight[1]};
  std::vector<int32_t> zero_point_bias = {0, 0};

  scales_zp = QuantizationParams<int8_t>(output_min, output_max);
  std::vector<float> scales_output = {scales_zp.first};
  std::vector<int32_t> zero_point_output = {scales_zp.second};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                     3, scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  nchw
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scales_input[0], zero_point_input[0]);

  // weight_data_float = {1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2};
  std::vector<int8_t> weight_data = {1, 3, 3, 5, 2, 4, 4, 6,
                                     4, 3, 2, 1, 4, 3, 2, 1};

  // bias_data_float ={3, -2};
  std::vector<int32_t> bias_data = {6, -2};

  // golden data
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {31, -57, 64, -46};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scales_output[0], zero_point_output[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_w_h_128_1_ksize_1_1_stride_2_int8_QuantizedPerChannelTest) {
  std::map<uint32_t, std::vector<uint32_t>> input_shape_list;
  input_shape_list[32] = {18, 20, 22, 26, 28, 30, 34, 36, 38,
                          42, 44, 46, 50, 52, 54, 58, 60, 62};
  input_shape_list[63] = {18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62};
  input_shape_list[95] = {18, 20, 22, 26, 28, 30, 34, 36, 38,
                          42, 44, 46, 50, 52, 54, 58, 60, 62};
  input_shape_list[96] = {18, 20, 22, 26, 28, 30, 34, 36, 38,
                          42, 44, 46, 50, 52, 54, 58, 60, 62};
  tim::vx::ShapeType input_shape({2, 2, 128, 1});     //whcn
  tim::vx::ShapeType weight_shape({1, 1, 128, 256});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[3], input_shape[3]});  //whcn

  std::vector<float> scales_input = {0.5};
  std::vector<int32_t> zero_point_input = {-1};
  std::vector<float> scales_weight(weight_shape[3]);
  std::vector<int32_t> zero_point_weight(weight_shape[3]);
  for (unsigned int i = 0; i < weight_shape[3]; i++) {
    scales_weight[i] = 1;
    zero_point_weight[i] = 0;
  }

  int32_t sizeofweight = scales_weight.size();
  std::vector<float> scales_bias(sizeofweight);
  std::vector<int32_t> zero_point_bias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scales_bias[i] = scales_input[0] * scales_weight[i];
    zero_point_bias[i] = 0;
  }

  std::vector<float> scales_output = {0.5};
  std::vector<int32_t> zero_point_output = {-1};

  tim::vx::Quantization quant_input(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scales_input, zero_point_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                     3, scales_weight, zero_point_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                   scales_bias, zero_point_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::ASYMMETRIC, 2,
                                     scales_output, zero_point_output);

  uint32_t weight_size =
      weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3];
  std::vector<float> weight_data_float(weight_size);
  for (uint32_t i = 0; i < weight_size; i++) {
    weight_data_float[i] = 1;
  }
  std::vector<int8_t> weight_data = Quantize<int8_t>(weight_data_float, 1, 0);

  // bias_data
  std::vector<int32_t> bias_data(weight_shape[3]);
  for (uint32_t i = 0; i < weight_shape[3]; i++) {
    bias_data[i] = 2;
  }

  for (std::map<uint32_t, std::vector<uint32_t>>::iterator iter =
           input_shape_list.begin();
       iter != input_shape_list.end(); iter++) {
    for (uint32_t j = 0; j < iter->second.size(); j++) {
      input_shape[0] = iter->first;
      input_shape[1] = iter->second[j];
      output_shape[0] = (input_shape[0] + 1) / 2;
      output_shape[1] = (input_shape[1] + 1) / 2;
      tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                     tim::vx::TensorAttribute::INPUT,
                                     quant_input);
      tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                      tim::vx::TensorAttribute::CONSTANT,
                                      quant_weight);
      tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                    tim::vx::TensorAttribute::CONSTANT,
                                    quant_bias);
      tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                      tim::vx::TensorAttribute::OUTPUT,
                                      quant_output);
      uint32_t input_size =
          input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
      std::vector<float> input_data_float(input_size);
      for (uint32_t i = 0; i < input_size; i++) {
        input_data_float[i] = 1;
      }
      std::vector<int8_t> input_data = Quantize<int8_t>(
          input_data_float, scales_input[0], zero_point_input[0]);

      uint32_t golden_size =
          output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
      std::vector<float> golden_float(golden_size);
      for (uint32_t i = 0; i < golden_size; i++) {
        golden_float[i] = 129;
      }
      std::vector<int8_t> golden = Quantize<int8_t>(
          golden_float, scales_output[0], zero_point_output[0]);

      auto ctx = tim::vx::Context::Create();
      auto graph = ctx->CreateGraph();
      auto input_tensor = graph->CreateTensor(input_spec);
      auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
      auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
      auto output_tensor = graph->CreateTensor(output_spec);

      auto padding = tim::vx::PadType::VALID;
      std::array<uint32_t, 2> stride({2, 2});
      std::array<uint32_t, 2> dilation({1, 1});

      auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
          padding, stride, dilation);
      (*conv2d)
          .BindInput(input_tensor)
          .BindInput(weight_tensor)
          .BindInput(bias_tensor)
          .BindOutput(output_tensor);

      EXPECT_TRUE(graph->Compile());

      input_tensor->CopyDataToTensor(input_data.data());

      EXPECT_TRUE(graph->Run());

      uint32_t output_size = 1;
      for (auto i : output_tensor->GetShape()) {
        output_size *= i;
      }
      std::vector<int8_t> output(output_size);
      EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
      EXPECT_EQ(golden, output);
    }
  }
}

TEST(Conv2d, shape_4_2_2_2_int16_DFPQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  if (ctx->isClOnly()) GTEST_SKIP();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({1, 1, 2, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn
  int8_t fl_input = 9, fl_weight = 8, fl_output = 8;
  tim::vx::Quantization quant_input(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                    fl_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                     fl_weight);
  tim::vx::Quantization quant_output(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                     fl_output);
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT16, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT16, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT16, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);

  // Input data  float
  std::vector<float> input_data_float = {
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data   float
  std::vector<float> weight_data_float = {
      1, 2  // first filter
  };
  //input data(dfp16)
  std::vector<int16_t> input_data = {256, 256, 256, 256,  512, 512, 512, 512,
                                     256, 256, 256, 256,  512, 512, 512, 512,
                                     256, 512, 768, 1024, 256, 512, 768, 1024,
                                     256, 512, 768, 1024, 256, 512, 768, 1024};
  //weight data(dfp16)
  std::vector<int16_t> weight_data = {256, 512};
  // bias data
  std::vector<int64_t> bias_data = {0};
  //golden
  std::vector<float> golden = {1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3,
                               1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6};

  auto input_tensor = graph->CreateTensor(input_spec, input_data.data());
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<int16_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  //transform output(int16) to fp
  std::vector<float> f;
  for (const auto& q : output) {
    f.push_back(q / (float)((int64_t)1 << fl_output));
  }
  EXPECT_EQ(golden, f);
}
TEST(Conv2d, shape_4_2_1_1_int16_DFPQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  if (ctx->isClOnly()) GTEST_SKIP();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn
  int8_t fl_input = 9, fl_weight = 8, fl_bias = 17, fl_output = 8;

  tim::vx::Quantization quant_input(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                    fl_input);
  tim::vx::Quantization quant_weight(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                     fl_weight);
  tim::vx::Quantization quant_bias(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                   fl_bias);
  tim::vx::Quantization quant_output(tim::vx::QuantType::DYNAMIC_FIXED_POINT,
                                     fl_output);
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT16, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quant_input);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT16, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quant_weight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT64, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quant_bias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT16, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quant_output);
  // Input data  nchw
  std::vector<float> input_data_float = {
      1, 1, 1, 1,  // row = 1
      2, 2, 3, 2   // row = 2
  };

  // weight data   oihw
  std::vector<float> weight_data_float = {
      1,  2,  3,  4,  //first 2x2 filter
      -1, 1,  -1, 1,  // second 2x2 filter
      -1, -1, 1,  1,  // third 2x2 filter
  };

  // bias data
  std::vector<float> bias_data_float = {1, 2, 3};

  // nchw
  std::vector<float> golden = {// first channel
                               18, 22, 21, 8, 7, 9, 8, 3,
                               // second channel
                               2, 3, 1, -1, 2, 3, 1, 0,
                               // third channel
                               5, 6, 6, 4, -1, -2, -2, 1};

  std::vector<int16_t> input_data = {512,  512,  512,  512,
                                     1024, 1024, 1536, 1024};
  std::vector<int16_t> weight_data = {256,  512, 768,  1024, -256, 256,
                                      -256, 256, -256, -256, 256,  256};
  std::vector<int64_t> bias_data = {1 << fl_bias, 2 * (1 << fl_bias),
                                    3 * (1 << fl_bias)};

  auto input_tensor = graph->CreateTensor(input_spec, input_data.data());
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<int16_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  //transform output(int16) to fp
  std::vector<float> f;
  for (const auto& q : output) {
    f.push_back(q / (float)((int64_t)1 << fl_output));
  }
  EXPECT_EQ(golden, f);
}

TEST(Conv2d, kernel_bigger_than_input_SAME) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 1, 1});   //whcn
  tim::vx::ShapeType kernel_shape({3, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({1});
  tim::vx::ShapeType output_shape({2, 3, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {
      1.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f,
  };
  std::vector<float> weight = {
      100.0f, 20.0f, 1.0f, 200.0f, 10.0f, 2.0f,
  };
  std::vector<float> bias = {500.0f};
  std::vector<float> golden = {
      567.0f, 1480.0f, 608.0f, 1370.0f, 543.0f, 760.0f,
  };
  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(kernel_spec, weight.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> dilations = {0, 0};
  std::array<uint32_t, 2> strides = {1, 1};
  auto op = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::SAME, strides, dilations, 0, tim::vx::DataLayout::WHCN,
      tim::vx::DataLayout::IcWHOc);
  (*op)
      .BindInputs({input_tensor, weight_tensor, bias_tensor})
      .BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}