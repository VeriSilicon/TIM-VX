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
#include "tim/vx/ops/nbg.h"

#include "gtest/gtest.h"

#include <vector>
#include <unordered_map>

TEST(graph, gen_binary_graph_with_empty_graph) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    size_t bin_size = -1;
    EXPECT_FALSE(graph->CompileToBinary(nullptr, &bin_size)) << "Can not generate binary graph if it is empty";
}

TEST(graph, gen_binary_graph_with_simple_add) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,1,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec);
    auto input_t1 = graph->CreateTensor(input_spec);
    auto output_t = graph->CreateTensor(output_spec);

    auto add = graph->CreateOperation<tim::vx::ops::Add>();
    (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    float in = 1.0f;
    float expected_out = 2.0f;
    EXPECT_TRUE(input_t0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

    EXPECT_TRUE(graph->Run());
    float output = 0.0f;
    EXPECT_TRUE(output_t->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);

    auto nbg_graph = ctx->CreateGraph();
    auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
    auto nbg_in1 = nbg_graph->CreateTensor(input_spec);
    auto nbg_out = nbg_graph->CreateTensor(output_spec);

    EXPECT_TRUE(nbg_in0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(nbg_in1->CopyDataToTensor(&in, sizeof(in)));

    auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
        (nbg_buf.data()), /*num_of_input*/ 2,
        /*num_of_output*/ 1);
    (*nbg_node).BindInputs({nbg_in0, nbg_in1}).BindOutputs({nbg_out});
    EXPECT_TRUE(nbg_graph->Compile());
    EXPECT_TRUE(nbg_graph->Run());

    output=0.0f;
    EXPECT_TRUE(nbg_out->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);
}

TEST(graph, G_3DUNET_ONNX_SOC) {
  std::shared_ptr<tim::vx::Tensor> dumpTensor;
  std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> dumpTensorMap;
  auto vx_context = tim::vx::Context::Create();
  auto vx_graph = vx_context->CreateGraph();
  tim::vx::TensorSpec vx_output_tspec;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {5, 7, 7, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[0] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {5, 7, 7, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[1] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {5, 7, 7, 64, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[2] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[3] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[4] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[5] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[6] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[7] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[8] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[9] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[10] = dumpTensor;

  std::vector<char> weights_11(4, 1);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({1})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_11.data());
  dumpTensorMap[11] = dumpTensor;

  std::vector<char> weights_12(128, 2);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_12.data());
  dumpTensorMap[12] = dumpTensor;

  std::vector<char> weights_13(128, 3);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_13.data());
  dumpTensorMap[13] = dumpTensor;

  std::vector<char> weights_14(221184, 3);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {3, 3, 3, 32, 64})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_14.data());
  dumpTensorMap[14] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 64})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[15] = dumpTensor;

  std::vector<char> weights_16(256, 5);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({64})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_16.data());
  dumpTensorMap[16] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[17] = dumpTensor;

  /* Node[0] */
  vx_graph->CreateOperation<tim::vx::ops::Div>()
      ->BindInputs({dumpTensorMap[7], dumpTensorMap[8]})
      .BindOutputs({dumpTensorMap[6]});
  /* Node[1] */
  vx_graph->CreateOperation<tim::vx::ops::Add>()
      ->BindInputs({dumpTensorMap[10], dumpTensorMap[11]})
      .BindOutputs({dumpTensorMap[9]});
  /* Node[2] */
  vx_graph->CreateOperation<tim::vx::ops::Sqrt>()
      ->BindInputs({dumpTensorMap[9]})
      .BindOutputs({dumpTensorMap[8]});
  /* Node[3] */
  vx_graph->CreateOperation<tim::vx::ops::Multiply>()
      ->BindInputs({dumpTensorMap[6], dumpTensorMap[12]})
      .BindOutputs({dumpTensorMap[5]});
  /* Node[4] */
  vx_graph->CreateOperation<tim::vx::ops::Add>()
      ->BindInputs({dumpTensorMap[5], dumpTensorMap[13]})
      .BindOutputs({dumpTensorMap[4]});
  /* Node[5] */
  vx_graph->CreateOperation<tim::vx::ops::LeakyRelu>(0.010000)
      ->BindInputs({dumpTensorMap[4]})
      .BindOutputs({dumpTensorMap[3]});
  /* Node[6] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Conv3d>(
          64, tim::vx::PadType::AUTO, std::array<int, 3ul>({3, 3, 3}),
          std::array<int, 3ul>({2, 2, 2}), std::array<int, 3ul>({1, 1, 1}),
          std::array<int, 6ul>({1, 1, 1, 1, 1, 1}), 0,
          tim::vx::DataLayout::WHDCN, tim::vx::DataLayout::WHDIcOc)
      ->BindInputs({dumpTensorMap[3], dumpTensorMap[14]})
      .BindOutputs({dumpTensorMap[2]});
  /* Node[7] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 64}))
      ->BindInputs({dumpTensorMap[16]})
      .BindOutputs({dumpTensorMap[15]});
  /* Node[8] */
  vx_graph->CreateOperation<tim::vx::ops::Add>()
      ->BindInputs({dumpTensorMap[2], dumpTensorMap[15]})
      .BindOutputs({dumpTensorMap[1]});
  /* Node[9] */
  vx_graph
      ->CreateOperation<tim::vx::ops::ReduceMean>(
          std::vector<int, std::allocator<int>>({0, 1, 2}), 1)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[17]});
  /* Node[10] */
  vx_graph->CreateOperation<tim::vx::ops::Sub>()
      ->BindInputs({dumpTensorMap[1], dumpTensorMap[17]})
      .BindOutputs({dumpTensorMap[0]});

  size_t bin_size = 0;
  std::vector<char> nbg_buf(bin_size);

  EXPECT_TRUE(vx_graph->CompileToBinary(nullptr, &bin_size));

  std::vector<char> tmp(bin_size);
  nbg_buf.swap(tmp);
  EXPECT_TRUE(vx_graph->CompileToBinary(nbg_buf.data(), &bin_size));

  auto vx_nbg_graph = vx_context->CreateGraph();
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors, output_tensors;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  std::vector<float> in_data_0(10 * 14 * 14 * 32 * 1, 1.5f);
  auto input_tensor_0 =
      vx_nbg_graph->CreateTensor(vx_output_tspec, in_data_0.data());
  input_tensors.push_back(input_tensor_0);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 32, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  std::vector<float> in_data_1(32, 2.5f);
  auto input_tensor_1 =
      vx_nbg_graph->CreateTensor(vx_output_tspec, in_data_1.data());
  input_tensors.push_back(input_tensor_1);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {5, 7, 7, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  auto output_tensor_0 = vx_nbg_graph->CreateTensor(vx_output_tspec);
  output_tensors.push_back(output_tensor_0);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {5, 7, 7, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  auto output_tensor_1 = vx_nbg_graph->CreateTensor(vx_output_tspec);
  output_tensors.push_back(output_tensor_1);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {10, 14, 14, 32, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  auto output_tensor_3 = vx_nbg_graph->CreateTensor(vx_output_tspec);
  output_tensors.push_back(output_tensor_3);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 1, 64, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  auto output_tensor_2 = vx_nbg_graph->CreateTensor(vx_output_tspec);
  output_tensors.push_back(output_tensor_2);

  auto nbg_node = vx_nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      nbg_buf.data(), input_tensors.size(), output_tensors.size());

  nbg_node->BindInputs(input_tensors).BindOutputs(output_tensors);
  EXPECT_TRUE(vx_nbg_graph->Compile());
  EXPECT_TRUE(vx_nbg_graph->Run());
}