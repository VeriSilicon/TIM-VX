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
#include "tim/vx/ops/space2batch.h"

#include "gtest/gtest.h"

TEST(Space2Batch, shape_2_2_3_1_fp32_whcn) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType in_shape({2, 2, 3, 1});
  tim::vx::ShapeType out_shape({1, 1, 3, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> in_data = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};
  std::vector<float> golden = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));

  auto op = graph->CreateOperation<tim::vx::ops::Space2Batch>(
      std::vector<int>{2, 2}, std::vector<int>{0, 0, 0, 0});

  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(12);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));

  EXPECT_EQ(golden, output);
}
