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
#include "tim/vx/ops/grucell.h"

#include "gtest/gtest.h"
#include "test_utils.h"

std::shared_ptr<tim::vx::Tensor> make_empty_tensor(
    std::shared_ptr<tim::vx::Graph> graph, const tim::vx::ShapeType& shape,
    const tim::vx::TensorAttribute& role);

TEST(GRUCell, unit_4) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  float tolerance = ctx->hasSP() ? 1e-4f : 1e-5f;

  uint32_t num_units = 2;
  uint32_t feature = 4;
  uint32_t batchs = 1;
  tim::vx::ShapeType in_shape({feature, batchs});
  tim::vx::ShapeType hstate_and_out_shape({num_units, batchs});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                                  hstate_and_out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  tim::vx::TensorSpec kernel_i_spec(tim::vx::DataType::FLOAT32,
                                    tim::vx::ShapeType({feature, num_units}),
                                    tim::vx::TensorAttribute::CONSTANT);

  tim::vx::TensorSpec kernel_r_spec(tim::vx::DataType::FLOAT32,
                                    tim::vx::ShapeType({num_units, num_units}),
                                    tim::vx::TensorAttribute::CONSTANT);

  std::vector<float> kernel_i2z = {-0.1201707124710083,  0.051147401332855225,
                                   -0.02161085605621338, 0.2582472562789917,
                                   -0.7641150951385498,  0.27272117137908936,
                                   0.4013441801071167,   -0.43467071652412415};
  std::vector<float> kernel_i2r = {-0.34522661566734314, 0.11888366937637329,
                                   0.6542353630065918,   0.6331415176391602,
                                   -0.2489457130432129,  -0.47332942485809326,
                                   -0.7532100081443787,  0.46069061756134033};
  std::vector<float> kernel_i2h = {
      -0.0012096166610717773, -0.05206263065338135, -0.418102502822876,
      -0.20800292491912842,   -0.5549647808074951,  -0.1337134838104248,
      0.14222955703735352,    -0.21347862482070923};
  std::vector<float> kernel_r2z = {-0.49559473991394043, -0.10428445041179657,
                                   0.39165210723876953, 0.38152191042900085};
  std::vector<float> kernel_r2r = {0.03387263044714928, -0.39444485306739807,
                                   0.4542817771434784, -0.4098765254020691};
  std::vector<float> kernel_r2h = {-0.5441233515739441, -0.35663682222366333,
                                   -0.3120974004268646, 0.6267299056053162};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto h_state_tensor = graph->CreateTensor(output_spec);

  auto kernel_i2z_tensor =
      graph->CreateTensor(kernel_i_spec, kernel_i2z.data());
  auto kernel_i2r_tensor =
      graph->CreateTensor(kernel_i_spec, kernel_i2r.data());
  auto kernel_i2h_tensor =
      graph->CreateTensor(kernel_i_spec, kernel_i2h.data());

  auto kernel_r2z_tensor =
      graph->CreateTensor(kernel_r_spec, kernel_r2z.data());
  auto kernel_r2r_tensor =
      graph->CreateTensor(kernel_r_spec, kernel_r2r.data());
  auto kernel_r2h_tensor =
      graph->CreateTensor(kernel_r_spec, kernel_r2h.data());

  std::vector<float> in_data = {1, 2, 3, 4};
  std::vector<float> golden = {-0.2719525, -0.5766771};

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  auto op = graph->CreateOperation<tim::vx::ops::GRUCell>(num_units);
  (*op)
      .BindInputs({
          input_tensor,
          make_empty_tensor(graph, tim::vx::ShapeType(hstate_and_out_shape),
                            tim::vx::TensorAttribute::INPUT),  //h_state
          kernel_i2z_tensor,                                   //KERNEL_I2
          kernel_i2r_tensor,                                   //KERNEL_I2
          kernel_i2h_tensor,                                   //KERNEL_I2
          kernel_r2z_tensor,                                   //KERNEL_R2
          kernel_r2r_tensor,                                   //KERNEL_R2
          kernel_r2h_tensor,                                   //KERNEL_R2
      })
      .BindOutputs({output_tensor, h_state_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, tolerance));
}
