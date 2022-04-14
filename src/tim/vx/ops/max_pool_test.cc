/****************************************************************************
*
*    Copyright (c) 2022 Vivante Corporation
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
#include "tim/vx/ops/pool2d.h"
#include <iostream>
#include "gtest/gtest.h"
#include "test_utils.h"

TEST(MAX, shape_6_6_1_1_fp32_kernel_3_stride_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({6, 6, 1, 1});
  tim::vx::ShapeType out_shape({3, 3, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      -1,  -2,  -3,  -4,  -5,  -5,
      -6,  -7,  -8,  -9,  -10, -10,
      -11, -12, -13, -14, -15, -15,
      -16, -17, -18, -19, -20, -20,
      -21, -22, -23, -24, -25, -20,
      -21, -22, -23, -24, -25, -20,
  };
  std::vector<float> golden = {
      -1, -3, -5,
      -11, -13, -15,
      -21, -23, -20,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  std::array<uint32_t, 4> pad = {0, 1, 0, 1};
  std::array<uint32_t, 2> ksize = {3, 3};
  std::array<uint32_t, 2> stride = {2, 2};
  auto round_type = tim::vx::RoundType::CEILING;
  auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, pad, ksize, stride, round_type);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}