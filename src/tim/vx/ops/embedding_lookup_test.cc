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
#include "tim/vx/ops/embedding_lookup.h"

#include "gtest/gtest.h"

TEST(EmbeddingLookup, shape_2_5_int32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType idx_shape({3});
  tim::vx::ShapeType lut_shape({2, 5});
  tim::vx::ShapeType out_shape({2, 3});

  tim::vx::TensorSpec idx_spec(tim::vx::DataType::INT32, idx_shape,
                              tim::vx::TensorAttribute::INPUT); 
  tim::vx::TensorSpec lut_spec(tim::vx::DataType::INT32, lut_shape,
                              tim::vx::TensorAttribute::INPUT);                        
  tim::vx::TensorSpec out_spec(tim::vx::DataType::INT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto idx_tensor = graph->CreateTensor(idx_spec);
  auto lut_tensor = graph->CreateTensor(lut_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<int32_t> idx_data = {0,3,4};
  std::vector<int32_t> lut_data = {
       1,2,3,4,5,6,7,8,9,10};
  std::vector<int32_t> golden = {
      1,2,7,8,9,10};

  EXPECT_TRUE(idx_tensor->CopyDataToTensor(idx_data.data(),
                                          idx_data.size() * sizeof(int32_t)));
  EXPECT_TRUE(lut_tensor->CopyDataToTensor(lut_data.data(),
                                          lut_data.size() * sizeof(int32_t)));
  auto op = graph->CreateOperation<tim::vx::ops::EmbeddingLookup>();
  (*op).BindInputs({idx_tensor,lut_tensor}).BindOutput(out_tensor);

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<int32_t> output(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}