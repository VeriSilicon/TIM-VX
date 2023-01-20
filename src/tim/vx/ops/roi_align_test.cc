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
#include "tim/vx/ops/roi_align.h"

#include "gtest/gtest.h"
#include "test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"

TEST(RoiAlign, shape_4_2_1_1_float32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  uint32_t height = 4;
  uint32_t width = 4;
  uint32_t channels = 1;
  uint32_t batch = 1;
  uint32_t num_rois = 4;
  uint32_t depth = channels;

  int32_t out_height = 2;
  int32_t out_width = 2;
  float height_ratio = 2.0f;
  float width_ratio = 2.0f;
  int32_t height_sample_num = 4;
  int32_t width_sample_num = 4;

  tim::vx::ShapeType input_shape({width, height, channels, batch});  //whcn
  tim::vx::ShapeType regions_shape({4, num_rois});
  tim::vx::ShapeType batch_index_shape({num_rois});
  tim::vx::ShapeType output_shape(
      {(uint32_t)out_width, (uint32_t)out_height, depth, num_rois});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec regions_spec(tim::vx::DataType::FLOAT32, regions_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec batch_index_spec(tim::vx::DataType::INT32,
                                       batch_index_shape,
                                       tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {-10.0f, -1.0f, 4.0f,  -5.0f, -8.0f, -2.0f,
                                   9.0f,   1.0f,  7.0f,  -2.0f, 3.0f,  -7.0f,
                                   -2.0f,  10.0f, -3.0f, 5.0f};

  std::vector<float> regions_data = {2.0f, 2.0f, 4.0f, 4.0f, 0.0f, 0.0f,
                                     8.0f, 8.0f, 2.0f, 0.0f, 4.0f, 8.0f,
                                     0.0f, 2.0f, 8.0f, 4.0f};

  std::vector<int32_t> batch_index_data = {0, 0, 0, 0};

  std::vector<float> golden = {
      0.375f, 5.125f, -0.375f, 2.875f, -0.5f,    -0.3125f, 3.1875f, 1.125f,
      0.25f,  4.25f,  4.875f,  0.625f, -0.1875f, 1.125f,   0.9375f, -2.625f};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto regions_tensor = graph->CreateTensor(regions_spec, regions_data.data());
  auto batch_index_tensor =
      graph->CreateTensor(batch_index_spec, batch_index_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto roi_align = graph->CreateOperation<tim::vx::ops::RoiAlign>(
      out_height, out_width, height_ratio, width_ratio, height_sample_num,
      width_sample_num);
  (*roi_align)
      .BindInput(input_tensor)
      .BindInput(regions_tensor)
      .BindInput(batch_index_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());
  regions_tensor->CopyDataToTensor(regions_data.data());
  batch_index_tensor->CopyDataToTensor(batch_index_data.data());

  EXPECT_TRUE(graph->Run());

  std::vector<float> output(num_rois * out_height * out_width * depth);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  ArraysMatch(golden, output, 1e-5f);
}