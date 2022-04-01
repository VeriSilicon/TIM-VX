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
#include "tim/vx/ops/dense.h"

#include "test_utils.h"
#include "gtest/gtest.h"

TEST(Dense, shape_2_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType weight_shape({2, 3});
  tim::vx::ShapeType bias_shape({3, 1});
  tim::vx::ShapeType out_shape({3, 2});
  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto in_tensor = graph->CreateTensor(in_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec);
  auto bias_tensor = graph->CreateTensor(bias_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {
      1,
      4,
      2,
      6,
  };
  std::vector<float> weight_data = {
      -3, 3, 2, 1, 0, 4,
  };
  std::vector<float> bias_data = {
      0.1,
      0.4,
      0.5,
  };
  std::vector<float> golden = {9.1, 6.4, 16.5, 12.1, 10.4, 24.5};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));
  EXPECT_TRUE(weight_tensor->CopyDataToTensor(
      weight_data.data(), weight_data.size() * sizeof(float)));
  EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias_data.data(),
                                            bias_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Dense>(0, 3);
  (*op)
      .BindInputs({in_tensor, weight_tensor, bias_tensor})
      .BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Dense, shape_1_2_3_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 1, 2, 3});   // (input_size, d1, d0, batch_size)
  tim::vx::ShapeType weight_shape({2, 3});     // (input_size, weights)
  tim::vx::ShapeType bias_shape({3, 1});       // (weights, 1)
  tim::vx::ShapeType out_shape({3, 1, 2, 3});  // (weights, d1, d0, batch_size)
  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto in_tensor = graph->CreateTensor(in_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec);
  auto bias_tensor = graph->CreateTensor(bias_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {
      0.12609188, 0.46347019, 0.89598465, 0.27901134, 0.35867718, 0.36897406,
      0.73463392, 0.27901134, 0.12609188, 0.46347019, 0.89598465, 0.27901134,
  };
  std::vector<float> weight_data = {
      -0.31930989, 0.37613347, 0.27901134, -1.36916667, 0.38031587, 0.21580373,
  };
  std::vector<float> bias_data = {
      0.12609188,
      0.46347019,
      0.21580373,
  };
  std::vector<float> golden = {
      0.260156, -0.135917, 0.363777, -0.0550594, 0.331447, 0.616773,
      0.150346, 0.0583582, 0.43184,  -0.0035385, 0.286428, 0.555408,
      0.260156, -0.135917, 0.363777, -0.0550594, 0.331447, 0.616773,
  };

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));
  EXPECT_TRUE(weight_tensor->CopyDataToTensor(
      weight_data.data(), weight_data.size() * sizeof(float)));
  EXPECT_TRUE(bias_tensor->CopyDataToTensor(bias_data.data(),
                                            bias_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Dense>(0, 3);
  (*op)
      .BindInputs({in_tensor, weight_tensor, bias_tensor})
      .BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}
