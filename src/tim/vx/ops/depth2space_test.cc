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
#include "tim/vx/ops/depth2space.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(DepthToSpace, shape_6_4_2_1_int32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({2, 1, 4, 1});
  tim::vx::ShapeType out_shape({4, 2, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      3, 5, 4, 8, 3, 5, 4, 8,
  };
  std::vector<float> golden = {
      3, 4, 5, 8, 3, 4, 5, 8,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::DepthToSpace>(
      2, tim::vx::ops::DepthToSpace::DCR_mode);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());
  std::vector<float> output(golden.size());

  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}