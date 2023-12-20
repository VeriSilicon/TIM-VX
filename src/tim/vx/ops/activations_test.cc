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
#include "tim/vx/ops/activations.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(Linear, shape_5_1_fp32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-2.5, -0.1, 0, 0.55,
                                std::numeric_limits<float>::infinity()};
  std::vector<float> golden = {-0.5, 1.9, 2, 2.55,
                               std::numeric_limits<float>::infinity()};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::Linear>(1, 2);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Linear, shape_5_1_fp32_omit_b) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-2.5, -0.1, 0, 0.55,
                                std::numeric_limits<float>::infinity()};
  std::vector<float> golden = {-5.0, -0.2, 0, 1.1,
                               std::numeric_limits<float>::infinity()};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::Linear>(2);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Gelu, shape_5_1_fp32_approximate) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({5, 1});
  tim::vx::ShapeType out_shape({5, 1});
  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {-3, -1, 0, 1, 3};
  std::vector<float> golden = {-0.00363752, -0.15880796, 0, 0.841192,
                               2.9963627};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Gelu>(true);
  (*op).BindInput(in_tensor).BindOutput(out_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Gelu, shape_5_1_uint8_Quantized) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({5, 1});
  tim::vx::ShapeType out_shape({5, 1});

  const float InputMin = -127, InputMax = 128, OutputMin = -127,
              OutputMax = 128;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};         //scale
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};  //zero point

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);

  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_float_data = {-3, -1, 0, 1, 3};
  std::vector<float> golden_float = {-0.00404951, -0.15865529, 0, 0.8413447,
                                     2.9959507};

  std::vector<uint8_t> input_data =
      Quantize<uint8_t>(in_float_data, scalesInput[0],
                        zeroPointsInput[0]);  //Quantification process
  std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::Gelu>(false);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<uint8_t> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, (uint8_t)1));
}

TEST(HardSigmoid, shape_5_1_uint8_Quantized) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({20, 1});
  tim::vx::ShapeType out_shape({20, 1});

  std::vector<float> scalesInput = {0.00228914};  //scale
  std::vector<int32_t> zeroPointsInput = {128};   //zero point

  std::vector<float> scalesOutput = {0.005};
  std::vector<int32_t> zeroPointsOutput = {128};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);

  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<uint8_t> in_data = {65,  255, 140, 92,  142, 122, 117,
                                  167, 132, 117, 44,  99,  109, 96,
                                  216, 222, 135, 126, 113, 100};
  std::vector<uint8_t> golden_data = {222, 240, 229, 225, 229, 227, 227,
                                      232, 228, 227, 220, 225, 226, 225,
                                      236, 237, 229, 228, 227, 225};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  auto op = graph->CreateOperation<tim::vx::ops::HardSigmoid>(0.2, 0.5);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<uint8_t> output(golden_data.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden_data, output, (uint8_t)1));
}

TEST(HardSigmoid, a_b_) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({20, 1});
  tim::vx::ShapeType out_shape({20, 1});

  std::vector<float> scalesInput = {0.00228914};  //scale
  std::vector<int32_t> zeroPointsInput = {128};   //zero point

  std::vector<float> scalesOutput = {0.005};
  std::vector<int32_t> zeroPointsOutput = {128};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);

  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<uint8_t> in_data = {65,  255, 140, 92,  142, 122, 117,
                                  167, 132, 117, 44,  99,  109, 96,
                                  216, 222, 135, 126, 113, 100};
  std::vector<uint8_t> golden_data = {239, 255, 250, 243, 250, 247, 246,
                                      253, 249, 246, 236, 244, 245, 244,
                                      255, 255, 249, 248, 246, 244};

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
  auto op = graph->CreateOperation<tim::vx::ops::HardSigmoid>(0.3, 0.6);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<uint8_t> output(golden_data.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden_data, output);
}

TEST(Elu, shape_5_1_fp32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-2.5, -0.1, 0, 0.55, 99};
  std::vector<float> golden = {-0.917915, -0.0951626, 0, 0.55, 99};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::Elu>();
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Elu, shape_5_1_fp32_a) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-2.5, -0.1, 0, 0.55, 99};
  std::vector<float> golden = {-0.458957, -0.0475813, 0, 0.55, 99};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::Elu>(0.5);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

#ifdef _VSI_NN_OP_SELU_H
TEST(Selu, shape_2_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2, 2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {2, 1, 3, 10};
  std::vector<float> golden = {2.1014, 1.0507, 3.1521, 10.507};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Selu>();
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}
#endif

#ifdef _VSI_NN_OP_CELU_H
TEST(Celu, shape_2_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 2});
  tim::vx::ShapeType out_shape({2, 2});

  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);

  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {-1, 0.71, 3, 10};
  std::vector<float> golden = {-0.69762, 0.71, 3, 10};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));

  auto op = graph->CreateOperation<tim::vx::ops::Celu>(1.3);
  (*op).BindInputs({in_tensor}).BindOutputs({out_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}
#endif

TEST(Sign, shape_5_1_fp32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-3.7, -1, 0, 0.5, 12};
  std::vector<float> golden = {-1, -1, 0, 1, 1};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::Sign>();
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(SoftSign, shape_5_1_fp32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({5, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {-3.7, -1, 0, 0.5, 12};
  std::vector<float> golden = {-0.78723, -0.5, 0, 0.33333, 0.92308};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::SoftSign>();
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(5, 0);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(HardSwish, 40_f32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  float tolerance = ctx->hasSP() ? 1e-3f : 1e-5f;

  tim::vx::ShapeType io_shape({40});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      4.53125f,   3.90625f,   3.046875f,  -8.59375f,  -1.328125f, 1.328125f,
      0.0f,       -8.515625f, -8.984375f, -0.234375f, 0.859375f,  9.84375f,
      -0.15625f,  -8.515625f, 8.671875f,  4.609375f,  9.21875f,   -1.796875f,
      1.171875f,  9.375f,     -8.75f,     2.421875f,  -8.125f,    -1.09375f,
      -9.609375f, -1.015625f, -9.84375f,  2.578125f,  4.921875f,  -5.078125f,
      5.0f,       -0.859375f, 1.953125f,  -6.640625f, -7.8125f,   4.453125f,
      -4.453125f, -6.875f,    0.78125f,   0.859375f};
  std::vector<float> golden = {
      4.53125f,    3.90625f,    3.046875f,   0.0f,        -0.3700765f,
      0.9580485f,  0.0f,        0.0f,        0.0f,        -0.1080322f,
      0.5527751f,  9.84375f,    -0.074056f,  0.0f,        8.671875f,
      4.609375f,   9.21875f,    -0.3603109f, 0.8148193f,  9.375f,
      0.0f,        2.1885173f,  0.0f,        -0.3474935f, 0.0f,
      -0.3358968f, 0.0f,        2.3968506f,  4.921875f,   0.0f,
      5.0f,        -0.3065999f, 1.6123454f,  0.0f,        0.0f,
      4.453125f,   0.0f,        0.0f,        0.4923503f,  0.5527751f};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto op = graph->CreateOperation<tim::vx::ops::HardSwish>();
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(40);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}