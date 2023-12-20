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
#include "tim/vx/ops.h"
#include "test_utils.h"
#include "gtest/gtest.h"
#include "tim/transform/layout_inference.h"
#include "permute_vector.h"

TEST(Prelu, alpha_boardcast) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2,2,3,1}); //input_pv={1,2,0,3}
  tim::vx::ShapeType alph_shape({3,1,1}); //pv={0,1,2}
  tim::vx::ShapeType output_shape({3,2,2,1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec alpha_spec(tim::vx::DataType::FLOAT32, alph_shape,
                                 tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                      tim::vx::TensorAttribute::OUTPUT);
  std::vector<float> in_data = {0.4, -0.2, 0.3, -0.4, 0.5, -0.6, 0.4, -0.2, 0.3, -0.4, 0.5, -0.6};
  std::vector<float> alpha_data = {1,2,3};
  std::vector<float> golden = {0.4, 0.5, 0.3, -0.2, -1.2, -1.2, 0.3, 0.4, 0.5, -0.4, -0.4, -1.8};

  auto input = graph->CreateTensor(input_spec);
  auto alpha = graph->CreateTensor(alpha_spec, alpha_data.data());
  auto output = graph->CreateTensor(output_spec);
  auto prelu = graph->CreateOperation<tim::vx::ops::Prelu>(0);
  (*prelu).BindInputs({input, alpha}).BindOutputs({output});

  std::map<std::shared_ptr<tim::vx::Tensor>,
           std::shared_ptr<tim::transform::IPermuteVector>>
      tensor_pv_map;
  std::shared_ptr<tim::transform::IPermuteVector> pv =
      std::make_shared<tim::transform::PermuteVector<4>>(
          std::initializer_list<uint32_t>({1,2,0,3}));
  tensor_pv_map.insert({input, pv});
  auto transform = tim::transform::LayoutInference(graph, ctx, tensor_pv_map);
  auto infer_graph = transform.first;
  EXPECT_TRUE(infer_graph->Compile());
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  EXPECT_TRUE(infer_input->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> out(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(out.data()));
  EXPECT_TRUE(ArraysMatch(golden, out, 1e-5f));
}


TEST(Prelu, alpha_as_input) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2,2,3,1}); //input_pv={1,2,0,3}
  tim::vx::ShapeType alph_shape({3,1,1}); //pv={0,1,2}
  tim::vx::ShapeType output_shape({3,2,2,1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec alpha_spec(tim::vx::DataType::FLOAT32, alph_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                      tim::vx::TensorAttribute::OUTPUT);
  std::vector<float> in_data = {0.4, -0.2, 0.3, -0.4, 0.5, -0.6, 0.4, -0.2, 0.3, -0.4, 0.5, -0.6};
  std::vector<float> alpha_data = {1,2,3};
  std::vector<float> golden = {0.4, 0.5, 0.3, -0.2, -1.2, -1.2, 0.3, 0.4, 0.5, -0.4, -0.4, -1.8};

  auto input = graph->CreateTensor(input_spec);
  auto alpha = graph->CreateTensor(alpha_spec);
  auto output = graph->CreateTensor(output_spec);
  auto prelu = graph->CreateOperation<tim::vx::ops::Prelu>(0);
  (*prelu).BindInputs({input, alpha}).BindOutputs({output});

  std::map<std::shared_ptr<tim::vx::Tensor>,
           std::shared_ptr<tim::transform::IPermuteVector>>
      tensor_pv_map;
  std::shared_ptr<tim::transform::IPermuteVector> pv =
      std::make_shared<tim::transform::PermuteVector<4>>(
          std::initializer_list<uint32_t>({1,2,0,3}));
  tensor_pv_map.insert({input, pv});
  auto transform = tim::transform::LayoutInference(graph, ctx, tensor_pv_map);
  auto infer_graph = transform.first;
  EXPECT_TRUE(infer_graph->Compile());
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_alpha = graph_io_map[graph->InputsTensor()[1]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  EXPECT_TRUE(infer_input->CopyDataToTensor(in_data.data(), in_data.size()));
  EXPECT_TRUE(infer_alpha->CopyDataToTensor(alpha_data.data(), alpha_data.size()));
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> out(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(out.data()));
  EXPECT_TRUE(ArraysMatch(golden, out, 1e-5f));
}