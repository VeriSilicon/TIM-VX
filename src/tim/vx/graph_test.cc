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

#include "gtest/gtest.h"

#include <vector>
#include <fstream>
#include <algorithm>
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

TEST(graph, gen_binary_graph_with_self_elementwise) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({1, 1, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, io_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_t0 = graph->CreateTensor(input_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Reshape>(io_shape);
  auto input_t1 = graph->CreateTensor(input_1_spec);
  (*op).BindInputs({input_t0}).BindOutputs({input_t1});

  auto output_t = graph->CreateTensor(output_spec);
  auto add = graph->CreateOperation<tim::vx::ops::Add>();
  (*add).BindInputs({input_t1, input_t1}).BindOutputs({output_t});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  std::cout << "CompileToBinary 1." << std::endl;
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));
  std::cout << "CompileToBinary 2." << std::endl;

  // binary graph compilation doesn't impact current graph's execution
  float in = 1.0f;
  float expected_out = 2.0f;
  EXPECT_TRUE(input_t0->CopyDataToTensor(&in, sizeof(in)));
  // EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

  EXPECT_TRUE(graph->Run());
  float output = 0.0f;
  EXPECT_TRUE(output_t->CopyDataFromTensor(&output));
  EXPECT_EQ(output, expected_out);
  std::cout << "graph->Run." << output << std::endl;

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
  auto nbg_out = nbg_graph->CreateTensor(output_spec);

  EXPECT_TRUE(nbg_in0->CopyDataToTensor(&in, sizeof(in)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_in0}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  output = 0.0f;
  EXPECT_TRUE(nbg_out->CopyDataFromTensor(&output));
  EXPECT_EQ(output, expected_out);
}

TEST(graph, gen_binary_graph_with_all_const_input) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({1, 1, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  float const_in = 6.0f;
  auto input_t0 = graph->CreateTensor(input_spec, &const_in);
  auto input_t1 = graph->CreateTensor(input_spec, &const_in);

  auto output_t = graph->CreateTensor(output_spec);
  auto add = graph->CreateOperation<tim::vx::ops::Add>();
  (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

  //   size_t bin_size = -1;
  //   EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  //   std::cout << "CompileToBinary 1." << std::endl;
  //   EXPECT_NE(bin_size, -1);
  //   std::vector<char> nbg_buf(bin_size);

  //   // generate binary graph does't require input data
  //   EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));
  //   std::cout << "CompileToBinary 2." << std::endl;

  EXPECT_TRUE(graph->Run());
  float output = 0.0f;
  float expected_out = 12.0f;
  EXPECT_TRUE(output_t->CopyDataFromTensor(&output));
  EXPECT_EQ(output, expected_out);
  std::cout << "graph->Run. " << output << std::endl;

  //   auto nbg_graph = ctx->CreateGraph();
  //   auto nbg_out = nbg_graph->CreateTensor(output_spec);

  //   auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
  //       (nbg_buf.data()), /*num_of_input*/ 0,
  //       /*num_of_output*/ 1);
  //   (*nbg_node).BindInputs({}).BindOutputs({nbg_out});
  //   EXPECT_TRUE(nbg_graph->Compile());
  //   EXPECT_TRUE(nbg_graph->Run());

  //   output = 0.0f;
  //   EXPECT_TRUE(nbg_out->CopyDataFromTensor(&output));
  //   EXPECT_EQ(output, expected_out);
}

TEST(graph, sqrt13) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({13, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, io_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_t0 = graph->CreateTensor(input_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Reshape>(io_shape);
  auto input_t1 = graph->CreateTensor(input_1_spec);
  (*op).BindInputs({input_t0}).BindOutputs({input_t1});

  auto output_t = graph->CreateTensor(output_spec);
  auto add = graph->CreateOperation<tim::vx::ops::Sqrt>();
  (*add).BindInputs({input_t1}).BindOutputs({output_t});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  std::cout << "CompileToBinary 1." << std::endl;
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));
  std::cout << "CompileToBinary 2." << std::endl;

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> in(13, 1.21f);
  std::vector<float> expected_out(13, 1.1f);
  EXPECT_TRUE(input_t0->CopyDataToTensor(in.data(), in.size() * sizeof(float)));
  // EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output(in.size());
  EXPECT_TRUE(output_t->CopyDataFromTensor(output.data()));
  EXPECT_EQ(output, expected_out);
  //   std::cout << "graph->Run." << output << std::endl;

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
  auto nbg_out = nbg_graph->CreateTensor(output_spec);

  EXPECT_TRUE(nbg_in0->CopyDataToTensor(in.data(), in.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_in0}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  //   output = 0.0f;
  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output.data()));
  EXPECT_EQ(output, expected_out);
}

TEST(graph, gen_binary_graph_with_self_elementwise_mul) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({13, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, io_shape,
                                   tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_t0 = graph->CreateTensor(input_spec);
  auto op = graph->CreateOperation<tim::vx::ops::Reshape>(io_shape);
  auto input_t1 = graph->CreateTensor(input_1_spec);
  (*op).BindInputs({input_t0}).BindOutputs({input_t1});

  auto output_t0 = graph->CreateTensor(input_1_spec);
  auto add = graph->CreateOperation<tim::vx::ops::Multiply>();
  (*add).BindInputs({input_t1, input_t1}).BindOutputs({output_t0});

  auto output_t = graph->CreateTensor(output_spec);
  auto reshape1 = graph->CreateOperation<tim::vx::ops::Reshape>(io_shape);
  (*reshape1).BindInputs({output_t0}).BindOutputs({output_t});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  std::cout << "CompileToBinary 1." << std::endl;
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));
  std::cout << "CompileToBinary 2." << std::endl;

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> in(13, 2.0f);
  std::vector<float> expected_out(13, 4.0f);
  EXPECT_TRUE(input_t0->CopyDataToTensor(in.data(), in.size() * sizeof(float)));
  // EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output(in.size());
  EXPECT_TRUE(output_t->CopyDataFromTensor(output.data()));
  EXPECT_EQ(output, expected_out);
  //   std::cout << "graph->Run." << output << std::endl;

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
  auto nbg_out = nbg_graph->CreateTensor(output_spec);

  EXPECT_TRUE(nbg_in0->CopyDataToTensor(in.data(), in.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_in0}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  //   output = 0.0f;
  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output.data()));
  EXPECT_EQ(output, expected_out);
}

TEST(graph, gen_binary_graph_with_big_batch) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({7, 7, 1, 2048});
  tim::vx::ShapeType kernel_shape({7, 7, 1, 512});
  tim::vx::ShapeType output_shape({1, 1, 512, 2048});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_t = graph->CreateTensor(input_spec);
  auto kernel_t = graph->CreateTensor(kernel_spec);
  auto output_t = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto op =
      graph->CreateOperation<tim::vx::ops::Conv2d>(padding, stride, dilation);
  (*op).BindInputs({input_t, kernel_t}).BindOutputs({output_t});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  std::cout << "CompileToBinary 1." << std::endl;
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));
  std::cout << "CompileToBinary 2." << std::endl;
}

// single channel single batch, without permute
TEST(graph, MPG6411) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({6, 4, 1, 1});
  tim::vx::ShapeType upd_shape({2, 2, 1, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor, upd_tensor}).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2,
                                 1, 5, 7, 5, 6,  1, 0, 6, 2, 7, 2, 8};
  std::vector<float> upd_data = {2, 6, 3, 1};
  std::vector<float> golden = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0,
                               0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  /* all cmodel execution failed here */
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

// multi channel single batch, without permute
TEST(graph, MPG6421) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({6, 4, 2, 1});
  tim::vx::ShapeType upd_shape({2, 2, 2, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor, upd_tensor}).BindOutputs({out_tensor});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8,
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8};
  std::vector<float> upd_data = {2, 6, 3, 1, 2, 6, 3, 1};
  std::vector<float> golden = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0,
                               0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0,
                               2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  /* all cmodel execution failed here */
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

// single channel single batch, with permute
TEST(graph, MPG1641T) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({1, 6, 4, 1});  // CWHN
  tim::vx::ShapeType upd_shape({1, 2, 2, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};  // CWHN -> WHCN
  std::vector<uint32_t> inp_shape_t(inp_perm.size());
  std::vector<uint32_t> upd_shape_t(inp_perm.size());
  for (uint32_t i = 0; i < inp_perm.size(); i++) {
    inp_shape_t[i] = inp_shape[inp_perm[i]];
    upd_shape_t[i] = upd_shape[inp_perm[i]];
  }
  tim::vx::TensorSpec inp_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec upd_spec_t(tim::vx::DataType::FLOAT32, upd_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);

  auto inp_tensor_t = graph->CreateTensor(inp_spec_t);
  auto upd_tensor_t = graph->CreateTensor(upd_spec_t);
  auto out_tensor_t = graph->CreateTensor(out_spec_t);

  auto inp_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  inp_permute->BindInput(inp_tensor).BindOutput(inp_tensor_t);
  auto upd_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  upd_permute->BindInput(upd_tensor).BindOutput(upd_tensor_t);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor_t, upd_tensor_t}).BindOutputs({out_tensor_t});

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(out_tensor_t).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2,
                                 1, 5, 7, 5, 6,  1, 0, 6, 2, 7, 2, 8};
  std::vector<float> upd_data = {2, 6, 3, 1};
  std::vector<float> golden = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0,
                               0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

// multi channel single batch, with permute
TEST(graph, MPG2641T) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({2, 6, 4, 1});  // CWHN
  tim::vx::ShapeType upd_shape({2, 2, 2, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};  // CWHN -> WHCN

  std::vector<uint32_t> inp_shape_t(inp_perm.size());
  std::vector<uint32_t> upd_shape_t(inp_perm.size());
  for (uint32_t i = 0; i < inp_perm.size(); i++) {
    inp_shape_t[i] = inp_shape[inp_perm[i]];
    upd_shape_t[i] = upd_shape[inp_perm[i]];
  }
  tim::vx::TensorSpec inp_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec upd_spec_t(tim::vx::DataType::FLOAT32, upd_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);

  auto inp_tensor_t = graph->CreateTensor(inp_spec_t);
  auto upd_tensor_t = graph->CreateTensor(upd_spec_t);
  auto out_tensor_t = graph->CreateTensor(out_spec_t);

  auto inp_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  inp_permute->BindInput(inp_tensor).BindOutput(inp_tensor_t);
  auto upd_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  upd_permute->BindInput(upd_tensor).BindOutput(upd_tensor_t);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor_t, upd_tensor_t}).BindOutputs({out_tensor_t});

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(out_tensor_t).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data_ = {7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2,
                                  1, 5, 7, 5, 6,  1, 0, 6, 2, 7, 2, 8};
  std::vector<float> inp_data;
  for (uint32_t i = 0; i < 2 * inp_data_.size(); i++) {
    inp_data.push_back(inp_data_[i / 2]);
  }

  std::vector<float> upd_data_ = {2, 6, 3, 1};
  std::vector<float> upd_data;
  for (uint32_t i = 0; i < 2 * upd_data_.size(); i++) {
    upd_data.push_back(upd_data_[i / 2]);
  }

  std::vector<float> golden_ = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0,
                                0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<float> golden;
  for (uint32_t i = 0; i < 2 * golden_.size(); i++) {
    golden.push_back(golden_[i / 2]);
  }
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

// single channel multi batch, with permute
TEST(graph, MPG1642T) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({1, 6, 4, 2});  // CWHN
  tim::vx::ShapeType upd_shape({1, 2, 2, 2});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};  // CWHN -> WHCN
  std::vector<uint32_t> inp_shape_t(inp_perm.size());
  std::vector<uint32_t> upd_shape_t(inp_perm.size());
  for (uint32_t i = 0; i < inp_perm.size(); i++) {
    inp_shape_t[i] = inp_shape[inp_perm[i]];
    upd_shape_t[i] = upd_shape[inp_perm[i]];
  }
  tim::vx::TensorSpec inp_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec upd_spec_t(tim::vx::DataType::FLOAT32, upd_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);

  auto inp_tensor_t = graph->CreateTensor(inp_spec_t);
  auto upd_tensor_t = graph->CreateTensor(upd_spec_t);
  auto out_tensor_t = graph->CreateTensor(out_spec_t);

  auto inp_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  inp_permute->BindInput(inp_tensor).BindOutput(inp_tensor_t);
  auto upd_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  upd_permute->BindInput(upd_tensor).BindOutput(upd_tensor_t);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor_t, upd_tensor_t}).BindOutputs({out_tensor_t});

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(out_tensor_t).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8,
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8};
  std::vector<float> upd_data = {2, 6, 3, 1, 2, 6, 3, 1};
  std::vector<float> golden = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0,
                               0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0,
                               2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

// multi channel multi batch, with permute
TEST(graph, MPG2642T) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({2, 6, 4, 2});  // CWHN
  tim::vx::ShapeType upd_shape({2, 2, 2, 2});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec upd_spec(tim::vx::DataType::FLOAT32, upd_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto upd_tensor = graph->CreateTensor(upd_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};  // CWHN -> WHCN
  std::vector<uint32_t> inp_shape_t(inp_perm.size());
  std::vector<uint32_t> upd_shape_t(inp_perm.size());
  for (uint32_t i = 0; i < inp_perm.size(); i++) {
    inp_shape_t[i] = inp_shape[inp_perm[i]];
    upd_shape_t[i] = upd_shape[inp_perm[i]];
  }
  tim::vx::TensorSpec inp_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec upd_spec_t(tim::vx::DataType::FLOAT32, upd_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec_t(tim::vx::DataType::FLOAT32, inp_shape_t,
                                 tim::vx::TensorAttribute::TRANSIENT);

  auto inp_tensor_t = graph->CreateTensor(inp_spec_t);
  auto upd_tensor_t = graph->CreateTensor(upd_spec_t);
  auto out_tensor_t = graph->CreateTensor(out_spec_t);

  auto inp_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  inp_permute->BindInput(inp_tensor).BindOutput(inp_tensor_t);
  auto upd_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  upd_permute->BindInput(upd_tensor).BindOutput(upd_tensor_t);

  std::array<uint32_t, 2> ksize = {3, 2};
  std::array<uint32_t, 2> stride = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::MaxpoolGrad>(
      tim::vx::PadType::VALID, ksize, stride);
  (*op).BindInputs({inp_tensor_t, upd_tensor_t}).BindOutputs({out_tensor_t});

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(out_tensor_t).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data_ = {
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8,
      7, 2, 5, 3, 10, 2, 3, 8, 9, 3, 4, 2, 1, 5, 7, 5, 6, 1, 0, 6, 2, 7, 2, 8};
  std::vector<float> inp_data;
  for (uint32_t i = 0; i < 2 * inp_data_.size(); i++) {
    inp_data.push_back(inp_data_[i / 2]);
  }

  std::vector<float> upd_data_ = {2, 6, 3, 1, 2, 6, 3, 1};
  std::vector<float> upd_data;
  for (uint32_t i = 0; i < 2 * upd_data_.size(); i++) {
    upd_data.push_back(upd_data_[i / 2]);
  }

  std::vector<float> golden_ = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0,
                                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 6, 0, 0, 0,
                                2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  std::vector<float> golden;
  for (uint32_t i = 0; i < 2 * golden_.size(); i++) {
    golden.push_back(golden_[i / 2]);
  }
  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));
  EXPECT_TRUE(upd_tensor->CopyDataToTensor(upd_data.data(),
                                           upd_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_upd = nbg_graph->CreateTensor(upd_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));
  EXPECT_TRUE(nbg_upd->CopyDataToTensor(upd_data.data(),
                                        upd_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 2,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp, nbg_upd}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(graph, PermuteTwice2641) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({2, 6, 4, 1});  // CWHN
  tim::vx::ShapeType mid_shape({6, 4, 2, 1});
  tim::vx::ShapeType out_shape({2, 6, 4, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec mid_spec(tim::vx::DataType::FLOAT32, mid_shape,
                               tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto mid_tensor = graph->CreateTensor(mid_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};
  auto in_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  in_permute->BindInput(inp_tensor).BindOutput(mid_tensor);

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(mid_tensor).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  std::vector<float> golden = inp_data;

  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(graph, PermuteTwice1642) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({1, 6, 4, 2});  // CWHN
  tim::vx::ShapeType mid_shape({6, 4, 1, 2});
  tim::vx::ShapeType out_shape({1, 6, 4, 2});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec mid_spec(tim::vx::DataType::FLOAT32, mid_shape,
                               tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto mid_tensor = graph->CreateTensor(mid_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};
  auto in_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  in_permute->BindInput(inp_tensor).BindOutput(mid_tensor);

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(mid_tensor).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  std::vector<float> golden = inp_data;

  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(graph, PermuteTwice1641) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({1, 6, 4, 1});  // CWHN
  tim::vx::ShapeType mid_shape({6, 4, 1, 1});
  tim::vx::ShapeType out_shape({1, 6, 4, 1});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec mid_spec(tim::vx::DataType::FLOAT32, mid_shape,
                               tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto mid_tensor = graph->CreateTensor(mid_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};
  auto in_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  in_permute->BindInput(inp_tensor).BindOutput(mid_tensor);

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(mid_tensor).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0,
                                 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  std::vector<float> golden = inp_data;

  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(graph, PermuteTwice2642) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType inp_shape({2, 6, 4, 2});  // CWHN
  tim::vx::ShapeType mid_shape({6, 4, 2, 2});
  tim::vx::ShapeType out_shape({2, 6, 4, 2});
  tim::vx::TensorSpec inp_spec(tim::vx::DataType::FLOAT32, inp_shape,
                               tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec mid_spec(tim::vx::DataType::FLOAT32, mid_shape,
                               tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);
  auto inp_tensor = graph->CreateTensor(inp_spec);
  auto mid_tensor = graph->CreateTensor(mid_spec);
  auto out_tensor = graph->CreateTensor(out_spec);

  std::vector<uint32_t> inp_perm = {1, 2, 0, 3};
  auto in_permute = graph->CreateOperation<tim::vx::ops::Transpose>(inp_perm);
  in_permute->BindInput(inp_tensor).BindOutput(mid_tensor);

  std::vector<uint32_t> out_perm = {2, 0, 1, 3};
  auto out_permute = graph->CreateOperation<tim::vx::ops::Transpose>(out_perm);
  out_permute->BindInput(mid_tensor).BindOutput(out_tensor);

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  // binary graph compilation doesn't impact current graph's execution
  std::vector<float> inp_data = {
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1};

  std::vector<float> golden = inp_data;

  EXPECT_TRUE(inp_tensor->CopyDataToTensor(inp_data.data(),
                                           inp_data.size() * sizeof(float)));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(out_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(inp_spec);
  auto nbg_out = nbg_graph->CreateTensor(out_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(inp_data.data(),
                                        inp_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(Broadcast, 3To32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3});
  tim::vx::ShapeType output_shape({3, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<uint32_t> shape = {3, 2};
  auto op = graph->CreateOperation<tim::vx::ops::Broadcast>(shape);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  size_t bin_size = -1;
  EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
  EXPECT_NE(bin_size, -1);
  std::vector<char> nbg_buf(bin_size);

  // generate binary graph does't require input data
  EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

  char file_name[256];
  sprintf(file_name, "bcast_gen_by_soc_exe_succ.nb");
  FILE* fp;
  fp = fopen(file_name, "wb");
  fwrite(nbg_buf.data(), bin_size, 1, fp);
  fclose(fp);
  printf("Save NBG in %s, nbg_bin_size = %d \n", file_name, (int)bin_size);

  std::vector<float> in_data = {
      1.f,
      2.f,
      3.f,
  };
  std::vector<float> golden = {1.f, 2.f, 3.f, 1.f, 2.f, 3.f};
  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));
  EXPECT_TRUE(graph->Run());

  std::vector<float> output_values(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);

  auto nbg_graph = ctx->CreateGraph();
  auto nbg_inp = nbg_graph->CreateTensor(input_spec);
  auto nbg_out = nbg_graph->CreateTensor(output_spec);

  EXPECT_TRUE(nbg_inp->CopyDataToTensor(in_data.data(),
                                        in_data.size() * sizeof(float)));

  auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      (nbg_buf.data()), /*num_of_input*/ 1,
      /*num_of_output*/ 1);
  (*nbg_node).BindInputs({nbg_inp}).BindOutputs({nbg_out});
  EXPECT_TRUE(nbg_graph->Compile());
  EXPECT_TRUE(nbg_graph->Run());

  EXPECT_TRUE(nbg_out->CopyDataFromTensor(output_values.data()));
  EXPECT_EQ(golden, output_values);
}

TEST(graph, G_SIM_DEFORM_DETR_R50_ONNX_SOC) {
  auto vx_context = tim::vx::Context::Create();
  auto vx_graph = vx_context->CreateGraph();
  tim::vx::TensorSpec vx_output_tspec;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({2, 4, 600, 8}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  auto dumpTensor_0 = vx_graph->CreateTensor(vx_output_tspec);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({2, 4, 600, 8, 1}),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  auto dumpTensor_1 = vx_graph->CreateTensor(vx_output_tspec);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({2, 4, 8, 600, 1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  auto dumpTensor_2 = vx_graph->CreateTensor(vx_output_tspec);

  std::vector<uint32_t> perm = {0, 1, 3, 2, 4};
  vx_graph->CreateOperation<tim::vx::ops::Transpose>(perm)
      ->BindInputs({dumpTensor_2})
      .BindOutputs({dumpTensor_1});

  std::vector<uint32_t> shape = {2, 4, 600, 8};
  vx_graph->CreateOperation<tim::vx::ops::Reshape>(shape)
      ->BindInputs({dumpTensor_1})
      .BindOutputs({dumpTensor_0});

  size_t bin_size = 0;
  std::vector<char> nbg_buf(bin_size);
  EXPECT_TRUE(vx_graph->CompileToBinary(nullptr, &bin_size));

  std::vector<char> tmp(bin_size);
  nbg_buf.swap(tmp);
  EXPECT_TRUE(vx_graph->CompileToBinary(nbg_buf.data(), &bin_size));

  std::vector<tim::vx::TensorSpec> input_specs_;
  std::vector<tim::vx::TensorSpec> output_specs_;

  auto vx_nbg_graph = vx_context->CreateGraph();
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({2, 4, 8, 600, 1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  input_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({2, 4, 600, 8}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  output_specs_.push_back(vx_output_tspec);

  auto nbg_node = vx_nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      nbg_buf.data(), input_specs_.size(), output_specs_.size());
  std::vector<float> in_data(2 * 4 * 8 * 600, 1.5f);
  std::vector<float> out_data(2 * 4 * 8 * 600);
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors, output_tensors;
  std::transform(input_specs_.begin(), input_specs_.end(),
                 std::back_inserter(input_tensors),
                 [vx_nbg_graph, &in_data](const tim::vx::TensorSpec& spec) {
                   auto input_tensor = vx_nbg_graph->CreateTensor(spec);
                   input_tensor->CopyDataToTensor(
                       in_data.data(), in_data.size() * sizeof(float));
                   return input_tensor;
                 });
  std::transform(output_specs_.begin(), output_specs_.end(),
                 std::back_inserter(output_tensors),
                 [vx_nbg_graph](const tim::vx::TensorSpec& spec) {
                   return vx_nbg_graph->CreateTensor(spec);
                 });

  nbg_node->BindInputs(input_tensors).BindOutputs(output_tensors);

  EXPECT_TRUE(vx_nbg_graph->Compile());
  EXPECT_TRUE(vx_nbg_graph->Run());
  EXPECT_TRUE(output_tensors[0]->CopyDataFromTensor(out_data.data()));
}

TEST(graph, G_YOLO3_10_ONNX_SOC) {
  std::shared_ptr<tim::vx::Tensor> dumpTensor;
  std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> dumpTensorMap;

  auto vx_context = tim::vx::Context::Create();
  auto vx_graph = vx_context->CreateGraph();
  tim::vx::TensorSpec vx_output_tspec;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[0] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[1] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[2] = dumpTensor;

  std::vector<char> weights_3 = {1};
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_3.data());
  dumpTensorMap[3] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[4] = dumpTensor;
  /* Node[0] */
  vx_graph->CreateOperation<tim::vx::ops::Equal>()
      ->BindInputs({dumpTensorMap[2], dumpTensorMap[3]})
      .BindOutputs({dumpTensorMap[1]});
  /* Node[1] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalAnd>()
      ->BindInputs({dumpTensorMap[1], dumpTensorMap[4]})
      .BindOutputs({dumpTensorMap[0]});

  size_t bin_size = 0;
  std::vector<char> nbg_buf(bin_size);
  EXPECT_TRUE(vx_graph->CompileToBinary(nullptr, &bin_size));

  std::vector<char> tmp(bin_size);
  nbg_buf.swap(tmp);
  EXPECT_TRUE(vx_graph->CompileToBinary(nbg_buf.data(), &bin_size));

  std::vector<tim::vx::TensorSpec> input_specs_;
  std::vector<tim::vx::TensorSpec> output_specs_;
  auto vx_nbg_graph = vx_context->CreateGraph();

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  input_specs_.push_back(vx_output_tspec);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  input_specs_.push_back(vx_output_tspec);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  output_specs_.push_back(vx_output_tspec);

  auto nbg_node = vx_nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      nbg_buf.data(), input_specs_.size(), output_specs_.size());

  std::vector<char> in_data = {0};
  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors, output_tensors;
  std::transform(input_specs_.begin(), input_specs_.end(),
                 std::back_inserter(input_tensors),
                 [vx_nbg_graph, &in_data](const tim::vx::TensorSpec& spec) {
                   auto input_tensor = vx_nbg_graph->CreateTensor(spec);
                   input_tensor->CopyDataToTensor(
                       in_data.data(), in_data.size() * sizeof(char));
                   return input_tensor;
                 });
  std::transform(output_specs_.begin(), output_specs_.end(),
                 std::back_inserter(output_tensors),
                 [vx_nbg_graph](const tim::vx::TensorSpec& spec) {
                   return vx_nbg_graph->CreateTensor(spec);
                 });
  nbg_node->BindInputs(input_tensors).BindOutputs(output_tensors);
  EXPECT_TRUE(vx_nbg_graph->Compile());
  EXPECT_TRUE(vx_nbg_graph->Run());
}

TEST(graph, G_VOXELNET_ONNX_SOC) {
  std::shared_ptr<tim::vx::Tensor> dumpTensor;
  std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> dumpTensorMap;
  auto vx_context = tim::vx::Context::Create();
  auto vx_graph = vx_context->CreateGraph();
  tim::vx::TensorSpec vx_output_tspec;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1, 35, 1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[0] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({35, 1}),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[1] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({35, 1}),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[2] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({35, 1}),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[3] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({7, 35, 1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[4] = dumpTensor;

  std::vector<char> weights_5(4, 1);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({1}),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_5.data());
  dumpTensorMap[5] = dumpTensor;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1, 35, 1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[6] = dumpTensor;

  /* Node[0] */
  std::vector<int> axis = {0};
  vx_graph->CreateOperation<tim::vx::ops::ReduceMax>(axis, 0)
      ->BindInputs({dumpTensorMap[4]})
      .BindOutputs({dumpTensorMap[3]});
  /* Node[1] */
  vx_graph->CreateOperation<tim::vx::ops::Equal>()
      ->BindInputs({dumpTensorMap[3], dumpTensorMap[5]})
      .BindOutputs({dumpTensorMap[2]});
  /* Node[2] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalNot>()
      ->BindInputs({dumpTensorMap[2]})
      .BindOutputs({dumpTensorMap[1]});
  /* Node[3] */
  std::vector<unsigned int> shape = {1, 35, 1};
  vx_graph->CreateOperation<tim::vx::ops::Reshape>(shape)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[0]});
  /* Node[4] */
  vx_graph->CreateOperation<tim::vx::ops::Reshape>(shape)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[6]});

  size_t bin_size = 0;
  std::vector<char> nbg_buf(bin_size);
  EXPECT_TRUE(vx_graph->CompileToBinary(nullptr, &bin_size));

  std::vector<char> tmp(bin_size);
  nbg_buf.swap(tmp);
  EXPECT_TRUE(vx_graph->CompileToBinary(nbg_buf.data(), &bin_size));

  std::vector<tim::vx::TensorSpec> input_specs_;
  std::vector<tim::vx::TensorSpec> output_specs_;
  auto vx_nbg_graph = vx_context->CreateGraph();
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32, tim::vx::ShapeType({7, 35, 1}),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));

  input_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1, 35, 1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));

  output_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8, tim::vx::ShapeType({1, 35, 1}),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1, {}, {}));

  output_specs_.push_back(vx_output_tspec);
  auto nbg_node = vx_nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      nbg_buf.data(), input_specs_.size(), output_specs_.size());

  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors, output_tensors;

  std::vector<float> in_data(7 * 5 * 1, 1.5f);
  std::transform(input_specs_.begin(), input_specs_.end(),
                 std::back_inserter(input_tensors),
                 [vx_nbg_graph, &in_data](const tim::vx::TensorSpec& spec) {
                   auto input_tensor = vx_nbg_graph->CreateTensor(spec);
                   input_tensor->CopyDataToTensor(
                       in_data.data(), in_data.size() * sizeof(float));
                   return input_tensor;
                 });
  std::transform(output_specs_.begin(), output_specs_.end(),
                 std::back_inserter(output_tensors),
                 [vx_nbg_graph](const tim::vx::TensorSpec& spec) {
                   return vx_nbg_graph->CreateTensor(spec);
                 });
  nbg_node->BindInputs(input_tensors).BindOutputs(output_tensors);
  EXPECT_TRUE(vx_nbg_graph->Compile());
  EXPECT_TRUE(vx_nbg_graph->Run());
}

TEST(graph, G_DETR3D_0913_NO_DCN_SIM_ONNX_SOC) {
  std::shared_ptr<tim::vx::Tensor> dumpTensor;
  std::unordered_map<uint32_t, std::shared_ptr<tim::vx::Tensor>> dumpTensorMap;
  auto vx_context = tim::vx::Context::Create();
  auto vx_graph = vx_context->CreateGraph();
  tim::vx::TensorSpec vx_output_tspec;

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
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
              {2, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[1] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 900, 6, 1})),
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
              {2, 900, 6, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[3] = dumpTensor;
  std::vector<char> weights_4(4, 1);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({1})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_4.data());
  dumpTensorMap[4] = dumpTensor;
  std::vector<char> weights_5(4, 2);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({1})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_5.data());
  dumpTensorMap[5] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[6] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[7] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[8] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {4, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[9] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {4, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[10] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[11] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[12] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 900, 1, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[13] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[14] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[15] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[16] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[17] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[18] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[19] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[20] = dumpTensor;
  std::vector<char> weights_21(4, 3);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({1})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_21.data());
  dumpTensorMap[21] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[22] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[23] = dumpTensor;
  std::vector<char> weights_24(4, 4);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>({1})),
      tim::vx::TensorAttribute::CONSTANT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec, weights_24.data());
  dumpTensorMap[24] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[25] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[26] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[27] = dumpTensor;
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::TRANSIENT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  dumpTensor = vx_graph->CreateTensor(vx_output_tspec);
  dumpTensorMap[28] = dumpTensor;
  /* Node[0] */
  vx_graph->CreateOperation<tim::vx::ops::Sub>()
      ->BindInputs({dumpTensorMap[3], dumpTensorMap[4]})
      .BindOutputs({dumpTensorMap[2]});
  /* Node[1] */
  vx_graph->CreateOperation<tim::vx::ops::Multiply>()
      ->BindInputs({dumpTensorMap[2], dumpTensorMap[5]})
      .BindOutputs({dumpTensorMap[1]});
  /* Node[2] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6}))
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[0]});
  /* Node[3] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6}))
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[6]});
  /* Node[4] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6}))
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[7]});
  /* Node[5] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6}))
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[8]});
  /* Node[6] */
  vx_graph->CreateOperation<tim::vx::ops::Multiply>()
      ->BindInputs({dumpTensorMap[10], dumpTensorMap[11]})
      .BindOutputs({dumpTensorMap[9]});
  /* Node[7] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalAnd>()
      ->BindInputs({dumpTensorMap[18], dumpTensorMap[19]})
      .BindOutputs({dumpTensorMap[17]});
  /* Node[8] */
  vx_graph
      ->CreateOperation<tim::vx::ops::StridedSlice>(
          std::vector<int, std::allocator<int>>({0, 0, 0, 0}),
          std::vector<int, std::allocator<int>>({1, 900, 6, 1}),
          std::vector<int, std::allocator<int>>({1, 1, 1, 1}), 0, 0, 0)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[20]});
  /* Node[9] */
  vx_graph->CreateOperation<tim::vx::ops::Greater>()
      ->BindInputs({dumpTensorMap[20], dumpTensorMap[21]})
      .BindOutputs({dumpTensorMap[19]});
  /* Node[10] */
  vx_graph
      ->CreateOperation<tim::vx::ops::StridedSlice>(
          std::vector<int, std::allocator<int>>({0, 0, 0, 0}),
          std::vector<int, std::allocator<int>>({1, 900, 6, 1}),
          std::vector<int, std::allocator<int>>({1, 1, 1, 1}), 0, 0, 0)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[23]});
  /* Node[11] */
  vx_graph->CreateOperation<tim::vx::ops::Less>()
      ->BindInputs({dumpTensorMap[23], dumpTensorMap[24]})
      .BindOutputs({dumpTensorMap[22]});
  /* Node[12] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalAnd>()
      ->BindInputs({dumpTensorMap[17], dumpTensorMap[22]})
      .BindOutputs({dumpTensorMap[16]});
  /* Node[13] */
  vx_graph
      ->CreateOperation<tim::vx::ops::StridedSlice>(
          std::vector<int, std::allocator<int>>({1, 0, 0, 0}),
          std::vector<int, std::allocator<int>>({2, 900, 6, 1}),
          std::vector<int, std::allocator<int>>({1, 1, 1, 1}), 0, 0, 0)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[26]});
  /* Node[14] */
  vx_graph->CreateOperation<tim::vx::ops::Greater>()
      ->BindInputs({dumpTensorMap[26], dumpTensorMap[21]})
      .BindOutputs({dumpTensorMap[25]});
  /* Node[15] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalAnd>()
      ->BindInputs({dumpTensorMap[16], dumpTensorMap[25]})
      .BindOutputs({dumpTensorMap[15]});
  /* Node[16] */
  vx_graph
      ->CreateOperation<tim::vx::ops::StridedSlice>(
          std::vector<int, std::allocator<int>>({1, 0, 0, 0}),
          std::vector<int, std::allocator<int>>({2, 900, 6, 1}),
          std::vector<int, std::allocator<int>>({1, 1, 1, 1}), 0, 0, 0)
      ->BindInputs({dumpTensorMap[1]})
      .BindOutputs({dumpTensorMap[28]});
  /* Node[17] */
  vx_graph->CreateOperation<tim::vx::ops::Less>()
      ->BindInputs({dumpTensorMap[28], dumpTensorMap[24]})
      .BindOutputs({dumpTensorMap[27]});
  /* Node[18] */
  vx_graph->CreateOperation<tim::vx::ops::LogicalAnd>()
      ->BindInputs({dumpTensorMap[15], dumpTensorMap[27]})
      .BindOutputs({dumpTensorMap[14]});
  /* Node[19] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Reshape>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 1, 900, 1, 6, 1}))
      ->BindInputs({dumpTensorMap[14]})
      .BindOutputs({dumpTensorMap[13]});
  /* Node[20] */
  vx_graph
      ->CreateOperation<tim::vx::ops::Transpose>(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {0, 1, 4, 2, 3, 5}))
      ->BindInputs({dumpTensorMap[13]})
      .BindOutputs({dumpTensorMap[12]});
  /* Node[21] */
  vx_graph->CreateOperation<tim::vx::ops::Cast>()
      ->BindInputs({dumpTensorMap[12]})
      .BindOutputs({dumpTensorMap[11]});

  size_t bin_size = 0;
  std::vector<char> nbg_buf(bin_size);
  EXPECT_TRUE(vx_graph->CompileToBinary(nullptr, &bin_size));

  std::vector<char> tmp(bin_size);
  nbg_buf.swap(tmp);
  EXPECT_TRUE(vx_graph->CompileToBinary(nbg_buf.data(), &bin_size));

  std::vector<std::pair<tim::vx::TensorSpec, std::vector<float>>> input_specs_;
  std::vector<tim::vx::TensorSpec> output_specs_;
  auto vx_nbg_graph = vx_context->CreateGraph();
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 900, 6, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  std::vector<float> in_data_0(2 * 900 * 6, 1.5f);
  input_specs_.push_back(std::make_pair(vx_output_tspec, in_data_0));
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {4, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  std::vector<float> in_data_1(4 * 6 * 900 * 6, 1.5f);
  input_specs_.push_back(std::make_pair(vx_output_tspec, in_data_1));
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::BOOL8,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {1, 900, 6, 1})),
      tim::vx::TensorAttribute::INPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  std::vector<float> in_data_2(900 * 6, 1.5f);
  input_specs_.push_back(std::make_pair(vx_output_tspec, in_data_2));

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  output_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  output_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  output_specs_.push_back(vx_output_tspec);

  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {2, 1, 900, 6})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  output_specs_.push_back(vx_output_tspec);
  vx_output_tspec = tim::vx::TensorSpec(
      tim::vx::DataType::FLOAT32,
      tim::vx::ShapeType(
          std::vector<unsigned int, std::allocator<unsigned int>>(
              {4, 1, 6, 900, 1, 1})),
      tim::vx::TensorAttribute::OUTPUT,
      tim::vx::Quantization(tim::vx::QuantType::NONE, -1,
                            std::vector<float, std::allocator<float>>({}),
                            std::vector<int, std::allocator<int>>({})));
  output_specs_.push_back(vx_output_tspec);
  auto nbg_node = vx_nbg_graph->CreateOperation<tim::vx::ops::NBG>(
      nbg_buf.data(), input_specs_.size(), output_specs_.size());

  std::vector<std::shared_ptr<tim::vx::Tensor>> input_tensors, output_tensors;
  std::transform(
      input_specs_.begin(), input_specs_.end(),
      std::back_inserter(input_tensors),
      [vx_nbg_graph](
          const std::pair<tim::vx::TensorSpec, std::vector<float>>& pair) {
        auto input_tensor = vx_nbg_graph->CreateTensor(pair.first);
        input_tensor->CopyDataToTensor(pair.second.data(),
                                       pair.second.size() * sizeof(float));
        return input_tensor;
      });
  std::transform(output_specs_.begin(), output_specs_.end(),
                 std::back_inserter(output_tensors),
                 [vx_nbg_graph](const tim::vx::TensorSpec& spec) {
                   return vx_nbg_graph->CreateTensor(spec);
                 });
  nbg_node->BindInputs(input_tensors).BindOutputs(output_tensors);
  EXPECT_TRUE(vx_nbg_graph->Compile());
  EXPECT_TRUE(vx_nbg_graph->Run());
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
// You can disable compile trace_test if only need replay
// #undef ENABLE_API_TRACE
#ifdef ENABLE_API_TRACE
#define API_TRACER_IMPLEMENTATION   // enable static members in api tracer
#define TARGET_NAMESPACE_NAME "tim::vx"
#include "tim/experimental/trace/trace_tvx.h"

namespace tvx = trace;

TEST(graph, trace_test) {
    // Replace all tim::vx name space with tvx, tvx can be alias of trace
    // namespace.
    auto ctx = tvx::Context::Create();
    auto graph = ctx->CreateGraph();

    tvx::ShapeType io_shape({1,2,2,1});
    tvx::TensorSpec input_spec(tvx::DataType::FLOAT32, io_shape, tvx::TensorAttribute::INPUT);
    tvx::TensorSpec output_spec(tvx::DataType::FLOAT32, io_shape, tvx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec);
    auto input_t1 = graph->CreateTensor(input_spec);
    auto input_t2 = graph->CreateTensor(input_spec);
    auto output_t0 = graph->CreateTensor(output_spec);

    auto reshape = graph->CreateOperation<tvx::ops::Reshape>(io_shape);
    (*reshape).BindInput(input_t0).BindOutput(input_t1);
    auto add = graph->CreateOperation<tvx::ops::Add>();
    (*add).BindInputs({input_t0, input_t2}).BindOutputs({output_t0});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    std::vector<float> in = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<float> expected_out = {2.2f, 4.4f, 6.6f, 8.8f};;
    EXPECT_TRUE(input_t0->CopyDataToTensor(in.data(), sizeof(float) * in.size()));
    EXPECT_TRUE(input_t2->CopyDataToTensor(in.data(), sizeof(float) * in.size()));

    EXPECT_TRUE(graph->Run());
    std::vector<float> output(in.size());
    EXPECT_TRUE(output_t0->CopyDataFromTensor(output.data()));
    EXPECT_EQ(output, expected_out);

    // extra test for Quantization apis
    tvx::Quantization quant0;
    quant0.SetType(tvx::QuantType::ASYMMETRIC);
    quant0.SetChannelDim(1);
    quant0.SetScales(std::vector<float>({0.2, 0.3}));
    quant0.SetZeroPoints(std::vector<int32_t>({2, 3}));

}
#endif /* #ifdef ENABLE_API_TRACE */


/*******************************************************************************
 * How to replay a trace_log.cc:
 * 1. Copy trace_log.cc in the root dir of tim-vx, rename with trace_log.rpl.cc
 * 2. And copy the trace_bin.bin file to the runtime workspace,
 *    rename with trace_bin.rpl.bin
 * 3. (optional) Add compile and run api call for specific graph
 * 4. Set follows 0->1 and re-compile.
 ******************************************************************************/
#if 0
#define API_REPLAYER_IMPLEMENTATION // enable static members in api replayer
#include "tim/experimental/trace/replayer.h"
TEST(graph, replay_test) {
    #include "trace_log.rpl.cc"
    // Manual compile and run the selected graph if those api calls not exist.
    // Like:
    // graph_12->Compile();
    // graph_12->Run();
    // Last rebuild unit-test and execute this case with:
    // `$build/install/bin/unit-test --gtest_filter=*replay_test*`
}

#endif /* #if 0 */
