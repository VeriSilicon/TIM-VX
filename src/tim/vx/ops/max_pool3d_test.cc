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
#include "tim/vx/ops/max_pool3d.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "gtest/gtest.h"

#ifdef VSI_FEAT_OP_MAX_POOL3D

TEST(MaxPool3d, shape_3_2_2_2_1_fp32_kernel_2_2_2_stride_1_1_1_VALID) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({3, 2, 2, 2, 1});//whdcn
  tim::vx::ShapeType out_shape({2, 1, 1, 2, 1});//whdcn
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      0, 1, 2,
      3, 4 ,5, // depth0 channel0
      6, 7, 8, 
      9, 10, 11, // depth1 channel0
      12, 13, 14,
      15, 16, 17,// depth0 channel1
      18, 19, 20,
      21, 22, 23 // depth1 channel1
  };
  std::vector<float> golden = {
      10,11,
      22,23
  };

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto round_type = tim::vx::RoundType::FLOOR;
  std::array<uint32_t, 3> ksize = {2, 2, 2}; //whd
  std::array<uint32_t, 3> stride = {1, 1, 1}; //whd
  std::array<uint32_t, 6> pad = {0, 0, 0, 0, 0, 0};
  
  auto op = graph->CreateOperation<tim::vx::ops::MaxPool3d>(
      round_type, ksize, stride, pad, tim::vx::PadType::VALID);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(MaxPool3d, shape_4_2_2_1_1_fp32_kernel_2_2_2_stride_1_1_1_SAME) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({4,2,2,1,1}); //whdcn
  tim::vx::ShapeType out_shape({4,2,2,1,1});  //whdcn
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      0, 6, 2, 4, 2, 5, 4, 3, 3, 2, 10, 7, 3, 2, 2, 4
  };
  std::vector<float> golden = {
     6, 10, 10, 7, 5, 5, 4, 4, 3, 10,
     10, 7, 3, 2, 4, 4
  }; 
  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));

  auto round_type = tim::vx::RoundType::FLOOR;
  std::array<uint32_t, 3> ksize = {2, 2, 2};
  std::array<uint32_t, 3> stride = {1, 1, 1};
  std::array<uint32_t, 6> pad = {0, 0, 0, 0, 0, 0}; 
  
  auto op = graph->CreateOperation<tim::vx::ops::MaxPool3d>(
      round_type, ksize, stride, pad, tim::vx::PadType::SAME);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

# endif //(VSI_FEAT_OP_MAX_POOL3D)
