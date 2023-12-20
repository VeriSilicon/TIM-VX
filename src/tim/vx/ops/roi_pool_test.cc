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
#include "tim/vx/ops/roi_pool.h"

#include "gtest/gtest.h"
#include "test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"

TEST(RoiPool, shape_4_2_1_1_float32) {
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
  float scale = 0.5f;


  tim::vx::ShapeType input_shape({width, height, channels, batch});  //whcn
  tim::vx::ShapeType regions_shape({5, num_rois});
  tim::vx::ShapeType output_shape(
      {(uint32_t)out_width, (uint32_t)out_height, depth, num_rois});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec regions_spec(tim::vx::DataType::FLOAT32, regions_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {-10.0f, -1.0f, 4.0f,  -5.0f,
                                  -8.0f, -2.0f, 9.0f,   1.0f,
                                   7.0f, -2.0f, 3.0f,  -7.0f,
                                   -2.0f,  10.0f, -3.0f, 5.0f};

  std::vector<float> regions_data = {0.0f, 2.0f, 2.0f, 4.0f, 4.0f,
                                     0.0f, 0.0f, 0.0f, 8.0f, 8.0f,
                                     0.0f, 2.0f, 0.0f, 4.0f, 8.0f,
                                     0.0f, 0.0f, 2.0f, 8.0f, 4.0f};

  std::vector<float> golden = {
      -2, 9, -2, 3,
      9, 9, 10, 5,
      -1, 9, 10, 3,
      9, 9, 7, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto regions_tensor = graph->CreateTensor(regions_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

   std::array<uint32_t, 2> size;
   size[0] = out_width;
   size[1] = out_height;
  auto roi_pool = graph->CreateOperation<tim::vx::ops::RoiPool>(tim::vx::PoolType::MAX, scale, size);
  (*roi_pool)
      .BindInput(input_tensor)
      .BindInput(regions_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()*sizeof(float)));
  EXPECT_TRUE(regions_tensor->CopyDataToTensor(regions_data.data(), regions_data.size()*sizeof(float)));
  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(num_rois * out_height * out_width * depth);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
