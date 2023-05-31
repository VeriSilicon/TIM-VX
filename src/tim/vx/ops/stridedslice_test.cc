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
#include "tim/vx/ops/stridedslice.h"
#include "tim/transform/layout_inference.h"

#include <array>
#include <algorithm>

#include "gtest/gtest.h"

TEST(StridedSlice, shape_) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  /* Using tensors of 5 dimensions. */
  static constexpr std::array<int, 5> BEGIN = {0, 0, 0, 0, 2};
  static constexpr std::array<int, 5> END = {1, 3, 10, 10, 4};
  static constexpr std::array<int, 5> STRIDES = {1, 1, 1, 1, 1};

  static constexpr int MASK_BEGIN = 0b11110;
  static constexpr int MASK_END = 0b11110;
  static constexpr int MASK_SHRINK = 0b00000;

  static constexpr std::array<size_t, 5> SHAPE_INPUT = {1, 3, 10, 10, 85};
  static constexpr std::array<size_t, 5> SHAPE_OUTPUT = {1, 3, 10, 10, 2};
  static constexpr size_t SLICE_AXIS = 4;

  static constexpr size_t LEN_DETECTION_FULL = 85;
  static constexpr size_t NUM_ELEMENTS_INPUT = 25500;  // 1 * 3 * 10 * 10 * 85
  static constexpr size_t NUM_ELEMENTS_OUTPUT = 600;   // 1 * 3 * 10 * 10 * 2
  static constexpr size_t NUM_DETECTIONS = 300;        // 1 * 3 * 10 * 10

  tim::vx::ShapeType vxShapeInput;
  tim::vx::ShapeType vxShapeOutput;

  std::reverse_copy(SHAPE_INPUT.cbegin(), SHAPE_INPUT.cend(),
                    std::back_inserter(vxShapeInput));
  std::reverse_copy(SHAPE_OUTPUT.cbegin(), SHAPE_OUTPUT.cend(),
                    std::back_inserter(vxShapeOutput));

  // Create TIM-VX tensors.
  auto specInput = tim::vx::TensorSpec(tim::vx::DataType::FLOAT32, vxShapeInput,
                                       tim::vx::TensorAttribute::INPUT);

  auto specOutput =
      tim::vx::TensorSpec(tim::vx::DataType::FLOAT32, vxShapeOutput,
                          tim::vx::TensorAttribute::OUTPUT);

  auto tensorInput = graph->CreateTensor(specInput);
  auto tensorOutput = graph->CreateTensor(specOutput);

  std::vector<int> begin;
  std::vector<int> end;
  std::vector<int> strides;

  std::reverse_copy(BEGIN.cbegin(), BEGIN.cend(), std::back_inserter(begin));
  std::reverse_copy(END.cbegin(), END.cend(), std::back_inserter(end));
  std::reverse_copy(STRIDES.cbegin(), STRIDES.cend(),
                    std::back_inserter(strides));
  auto opStridedSlice = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);

  opStridedSlice->BindInput(tensorInput);
  opStridedSlice->BindOutput(tensorOutput);

  // Compile graph.
  bool ret = false;
  ret = graph->Compile();
  EXPECT_TRUE(ret) << "Compile Graph Failed";

  std::array<float, NUM_ELEMENTS_INPUT> bufferInput;
  std::array<float, NUM_ELEMENTS_OUTPUT> bufferOutput;

  // Prepare input tensor data.
  bufferInput.fill(0.0F);
  for (size_t k = 0; k < NUM_DETECTIONS; k++) {
    float* dataPtr = bufferInput.data() + k * LEN_DETECTION_FULL;
    for (auto i = BEGIN[SLICE_AXIS]; i < END[SLICE_AXIS]; i++) {
      dataPtr[i] = static_cast<float>(i);
    }
  }

  // Run graph.
  ret = tensorInput->CopyDataToTensor(bufferInput.data());
  ret = graph->Run();
  ret = tensorOutput->CopyDataFromTensor(bufferOutput.data());

  EXPECT_TRUE(ret) << "Failed at execute";
}

TEST(StridedSlice, shrinkmask_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 1});
  tim::vx::ShapeType output_shape({3, 2});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {1, 1, 2, 2, 3, 3};
  std::vector<float> golden = {1, 1, 2, 2, 3, 3};

  std::vector<int> begin = {0, 0, 0};
  std::vector<int> end = {0, 0, 0};
  std::vector<int> strides = {1, 1, 1};
  // The ith bits in MASK_SHRINK will mask input_shape[i].
  uint32_t MASK_BEGIN = 0, MASK_END = 0, MASK_SHRINK = 0b100;

  auto op = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  input_tensor->CopyDataToTensor(in_data.data(),
                                 in_data.size() * sizeof(float));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(StridedSlice, endmask_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 1});
  tim::vx::ShapeType output_shape({3, 2, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {1, 1, 2, 2, 3, 3};
  std::vector<float> golden = {1, 1, 2, 2, 3, 3};

  std::vector<int> begin = {2, 0, 0};
  std::vector<int> end = {3, 2, 1};
  std::vector<int> strides = {1, 1, 1};
  // The ith bits in MASK_BEGIN will mask begin[i]. MASK_END is similar.
  uint32_t MASK_BEGIN = 0b001, MASK_END = 0, MASK_SHRINK = 0;

  auto op = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  input_tensor->CopyDataToTensor(in_data.data(),
                                 in_data.size() * sizeof(float));

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}