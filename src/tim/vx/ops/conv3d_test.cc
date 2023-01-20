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
#include "tim/vx/ops/conv3d.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"
#include "test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"

TEST(Conv3d, shape_1_1_2_3_3_float32_simple_whdcn) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 2, 1, 1});   //whdcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1, 2});  //whdIcOc
  tim::vx::ShapeType output_shape(
      {2, 2, 2, weight_shape[4], input_shape[4]});  //whdcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {
      0.222290,  -0.735840, -2.349609, 1.327148,  0.645020,  0.059631,
      -1.081055, -1.307617, -0.306641, -0.520996, 0.041046,  3.234375,
      -2.269531, -2.121094, 1.269531,  -0.593750, -1.734375, -2.640625};

  std::vector<float> weight_data = {1.345703, 1.777344,  -1.022461, -1.070312,
                                    1.372070, -0.918945, 0.480713,  1.415039};

  // whdcn
  std::vector<float> golden = {-3.056641, -5.890625, 5.437500,  2.638672,
                               3.962891,  6.613281,  -4.359375, 4.000000,
                               2.531250,  1.543945,  -1.141602, -0.232300,
                               -4.843750, -2.138672, -3.904297, -8.648438};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<int32_t, 6> padding ({0, 0, 0, 0, 0, 0});
  std::array<int32_t, 3> stride({1, 1, 1});
  std::array<int32_t, 3> dilation({1, 1, 1});

  auto conv3d = graph->CreateOperation<tim::vx::ops::Conv3d>(
      padding, stride, dilation);
  (*conv3d)
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
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  for (uint32_t idx = 0; idx < golden.size(); idx++) {
      EXPECT_TRUE(std::abs(golden[idx] - output[idx]) < 0.01);
  }
}

TEST(Conv3d, shape_1_1_2_3_3_float32_simple_cwhdn) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 3, 2, 1});   //cwhdn
  tim::vx::ShapeType weight_shape({2, 1, 2, 2, 1});  //OcIcWHD
  tim::vx::ShapeType output_shape(
      {weight_shape[0], 2, 2, 2, input_shape[4]});  //cwhdn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {
      0.97471274, 0.76463452, 0.86721926, 0.92130888, 0.03260213, 0.08942557,
      0.44689693, 0.97484119, 0.55602722, 0.82500644, 0.9202445,  0.37466433,
      0.91804717, 0.56083073, 0.98317178, 0.60991722, 0.39409797, 0.40177473};

  std::vector<float> weight_data = {0.88074152, 0.43367621, 0.74519104,
                                    0.30248252, 0.93564262, 0.78602735,
                                    0.66508319, 0.84253425};

  std::vector<float> golden = {2.3119678, 1.4056407, 1.4096688, 0.69489276,
                               1.902216,  1.5820216, 1.3772604, 1.2759123,
                               2.6443386, 1.8302729, 2.268322,  1.7816017,
                               2.0592608, 1.3792293, 1.8625461, 1.1888919};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<int32_t, 6> padding ({0, 0, 0, 0, 0, 0});
  std::array<int32_t, 3> stride({1, 1, 1});
  std::array<int32_t, 3> dilation({1, 1, 1});

  auto conv3d = graph->CreateOperation<tim::vx::ops::Conv3d>(
      padding, stride, dilation, 0, tim::vx::DataLayout::CWHDN, tim::vx::DataLayout::OcIcWHD);
  (*conv3d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindOutput(output_tensor);

  auto final_graph = tim::transform::LayoutInference(graph, ctx);

  EXPECT_TRUE(final_graph.first->Compile());

  final_graph.second[input_tensor]->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(final_graph.first->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(final_graph.second[output_tensor]->CopyDataFromTensor(output.data()));
  for (uint32_t idx = 0; idx < golden.size(); idx++) {
      EXPECT_TRUE(std::abs(golden[idx] - output[idx]) < 0.01);
  }
}

TEST(Conv3d, DISABLED_shape_4_2_2_2_1_float32_simple_whdcn) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2, 1});   //whdcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 2, 2});  //whdIcOc
  tim::vx::ShapeType output_shape(
      {3, 1, 1, weight_shape[4], input_shape[4]});  //whdcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

  std::vector<float> weight_data = {-1, -1, -1, -1, -1, 1, -1, 1, -1, 1,  1,  1, 1, 1,  -1, -1,
               1,  -1, 1,  1,  1,  1, -1, 1, -1, -1, -1, 1, 1, -1, 1,  -1};

  // whdcn
  std::vector<float> golden = {26, 24, 22, -8, -6, -4};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<int32_t, 3> stride({1, 1, 1});
  std::array<int32_t, 3> dilation({1, 1, 1});

  auto conv3d = graph->CreateOperation<tim::vx::ops::Conv3d>(
      tim::vx::PadType::VALID, stride, dilation);
  (*conv3d)
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
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(output,golden);
}