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
#include "tim/vx/ops/erf.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(Erf, shape_3_2_fp32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2});
  tim::vx::ShapeType out_shape({3, 2});
  tim::vx::TensorSpec in_spec(tim::vx::DataType::FLOAT32, in_shape,
                              tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto in_tensor = graph->CreateTensor(in_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<float> in_data = {
       1, 2, 3,
       0,-1,-2};
  std::vector<float> golden = {
      0.8427007, 0.9953223, 0.999978,
      0        ,-0.8427007,-0.9953223};

  EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(),
                                          in_data.size() * sizeof(float)));
  auto op = graph->CreateOperation<tim::vx::ops::Erf>();
  (*op).BindInput(in_tensor).BindOutput(out_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-2f));
}

TEST(Erf, shape_3_2_uint8_Quantized) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2});
  tim::vx::ShapeType out_shape({3, 2});

  const float InputMin = -128, InputMax = 127, OutputMin = -128,
              OutputMax = 127;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};         //scale
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};  //zero point

  scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
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

  std::vector<float> in_data_float = {
       1, 2, 3,
       0,-1,-2};
  std::vector<float> golden_float = {
      0.8427007, 0.9953223, 0.999978,
      0        ,-0.8427007,-0.9953223};

  std::vector<uint8_t> input_data =
      Quantize<uint8_t>(in_data_float, scalesInput[0],
                        zeroPointsInput[0]);  //Quantification process
  std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::Erf>();
  (*op).BindInput(input_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<uint8_t> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}