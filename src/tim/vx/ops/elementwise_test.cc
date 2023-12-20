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
#include "tim/vx/ops/elementwise.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(FloorDiv, shape_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    if (ctx->hasSP()) GTEST_SKIP();

    tim::vx::ShapeType io_shape({1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = { 1 };
    std::vector<float> in_data_y = { 0 };
    std::vector<float> golden = { std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(1);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(FloorDiv, shape_5_1_broadcast_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({5, 1});
    tim::vx::ShapeType in_shape_y({1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::FLOAT32,
                            in_shape_x, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::FLOAT32,
                            in_shape_y, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = { 1, 3, -2, 0, 99 };
    std::vector<float> in_data_y = { 2 };
    std::vector<float> golden = { 0, 1, -1, 0, 49 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}


TEST(FloorDiv, shape_5_1_broadcast_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({1});
    tim::vx::ShapeType in_shape_y({5, 1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::UINT8,
                            in_shape_x, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::UINT8,
                            in_shape_y, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, quant_out);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data_x = { 255 };
    std::vector<uint8_t> in_data_y = { 1, 3, 2, 0, 255 };
    std::vector<uint8_t> golden = { 255, 170, 254, 255, 2 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()));
    auto op = graph->CreateOperation<tim::vx::ops::FloorDiv>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Div, shape_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    if (ctx->hasSP()) GTEST_SKIP();

    tim::vx::ShapeType io_shape({1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = { 1 };
    std::vector<float> in_data_y = { 0 };
    std::vector<float> golden = { std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Div>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(1);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Div, shape_5_1_broadcast_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({1});
    tim::vx::ShapeType in_shape_y({5, 1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::UINT8,
                            in_shape_x, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::UINT8,
                            in_shape_y, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, quant_out);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data_x = { 255 };
    std::vector<uint8_t> in_data_y = { 1, 2, 3, 0, 255 };
    std::vector<uint8_t> golden = { 255, 255, 170, 255, 2 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()));
    auto op = graph->CreateOperation<tim::vx::ops::Div>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Div, shape_5_1_broadcast_scale_uint8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({1});
    tim::vx::ShapeType in_shape_y({5, 1});
    tim::vx::ShapeType out_shape({5, 1});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 1, 0);
    tim::vx::Quantization quant_out(tim::vx::QuantType::ASYMMETRIC, 0.5, 0);
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::UINT8,
                            in_shape_x, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::UINT8,
                            in_shape_y, tim::vx::TensorAttribute::INPUT, quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT, quant_out);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data_x = { 128 };
    std::vector<uint8_t> in_data_y = { 1, 2, 3, 0, 255 };
    std::vector<uint8_t> golden = { 128, 64, 43, 255, 1 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()));
    auto op = graph->CreateOperation<tim::vx::ops::Div>(0.5f);
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(5);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, (uint8_t)1));
}

TEST(Div, Div_uint8) {
  auto context = tim::vx::Context::Create();
  auto graph = context->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 1, 1});
  tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 1.0, 0);
  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, input_quant);
  uint8_t data1[] = {1, 2, 3, 4, 5, 6};
  uint8_t data2[] = {2, 2, 2, 2, 2, 2};
  auto input1 = graph->CreateTensor(input_spec, data1);
  auto input2 = graph->CreateTensor(input_spec, data2);

  tim::vx::ShapeType output_shape({2, 3, 1, 1});
  tim::vx::Quantization output_quant(tim::vx::QuantType::ASYMMETRIC, 1.0, 0);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  output_quant);
  auto output = graph->CreateTensor(output_spec);

  auto op = graph->CreateOperation<tim::vx::ops::Div>();
  (*op).BindInputs({input1, input2}).BindOutputs({output});

  if (!graph->Compile()) {
    std::cout << "Compile graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  graph->PrintGraph();

  if (!graph->Run()) {
    std::cout << "Run graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  std::vector<uint8_t> output_data;
  std::vector<uint8_t> golden={0,1,2,2,2,3,0,0,0,0};
  output_data.resize(1 * 10);
  if (!output->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  EXPECT_TRUE(ArraysMatch(golden, output_data, (uint8_t)1));
}

TEST(Div, DISABLED_Div_int32) {
  auto context = tim::vx::Context::Create();
  auto graph = context->CreateGraph();

  tim::vx::ShapeType input_shape({1, 2, 2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  int32_t data1[] = {-2, 2, -15, 8};
  int32_t data2[] = {5, -2, -3, 5};
  auto input1 = graph->CreateTensor(input_spec, data1);
  auto input2 = graph->CreateTensor(input_spec, data2);

  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, input_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output = graph->CreateTensor(output_spec);

  auto op = graph->CreateOperation<tim::vx::ops::Div>();
  (*op).BindInputs({input1, input2}).BindOutputs({output});

  if (!graph->Compile()) {
    std::cout << "Compile graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  graph->PrintGraph();

  if (!graph->Run()) {
    std::cout << "Run graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  std::vector<int32_t> output_data;
  std::vector<int32_t> golden = {0, -1, 5, 2};
  output_data.resize(golden.size());
  if (!output->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
    EXPECT_TRUE(-1);
  }
  // div can have an error of 1 according to different rounding rules
  EXPECT_TRUE(ArraysMatch(golden, output_data, 1));
}

TEST(Div, DISABLED_Div_int32_broadcast) {
  auto context = tim::vx::Context::Create();
  auto graph = context->CreateGraph();

  tim::vx::ShapeType input1_shape({2,2,1,2,1});
  tim::vx::ShapeType input2_shape({1});
  tim::vx::TensorSpec input1_spec(tim::vx::DataType::INT32, input1_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input2_spec(tim::vx::DataType::INT32, input2_shape,
                                 tim::vx::TensorAttribute::INPUT);
  int32_t data1[] = {-20, 21, 7, 8, 11, -123, -42, -48};
  int32_t data2[] = {3};
  auto input1 = graph->CreateTensor(input1_spec, data1);
  auto input2 = graph->CreateTensor(input2_spec, data2);

  tim::vx::ShapeType output_shape({2,2,1,2,1});
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output = graph->CreateTensor(output_spec);

  auto op = graph->CreateOperation<tim::vx::ops::Div>();
  (*op).BindInputs({input1, input2}).BindOutputs({output});

  if (!graph->Compile()) {
    std::cout << "Compile graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  graph->PrintGraph();

  if (!graph->Run()) {
    std::cout << "Run graph fail." << std::endl;
    EXPECT_TRUE(-1);
  }

  std::vector<int32_t> output_data;
  std::vector<int32_t> golden = {-7, 7, 2, 3, 4, -41, -14, -16};
  output_data.resize(golden.size());
  if (!output->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
    EXPECT_TRUE(-1);
  }
  // div can have an error of 1 according to different rounding rules
  EXPECT_TRUE(ArraysMatch(golden, output_data, 1));
}

TEST(Minimum, shape_1_1_2_1_1_3_broadcast_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape_x({1, 1, 2, 1, 3});
    tim::vx::ShapeType in_shape_y({1});
    tim::vx::ShapeType out_shape({1, 1, 2, 1, 3});
    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::INT32,
                            in_shape_x, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::INT32,
                            in_shape_y, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int> in_data_x = { 1, 0, -1, -2, 3, 11 };
    std::vector<int> in_data_y = { 2 };
    std::vector<int> golden = { 1, 0, -1, -2, 2, 2 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Minimum>();
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int> output(golden.size());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}