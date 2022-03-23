/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"

TEST(Stack, shape_2_3_axis_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1,4,
        2,5,
        3,6,

        1,4,
        2,5,
        3,6
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(2, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Stack, shape_2_3_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1,4,
        1,4,
        2,5,

        2,5,
        3,6,
        3,6,
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(1, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Stack, shape_2_3_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({2,3});
    tim::vx::ShapeType output_shape({2,3,2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor1 = graph->CreateTensor(input_spec);
    auto input_tensor2 = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data1 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> in_data2 = {
        1,4,
        2,5,
        3,6
    };
    std::vector<float> golden = {
        1, 1,
        4, 4,
        2, 2,

        5, 5,
        3, 3,
        6, 6
        };

    EXPECT_TRUE(input_tensor1->CopyDataToTensor(
        in_data1.data(), in_data1.size() * sizeof(float)));
    EXPECT_TRUE(input_tensor2->CopyDataToTensor(
        in_data2.data(), in_data2.size() * sizeof(float)));
    auto op = graph->CreateOperation<tim::vx::ops::Stack>(0, 2);
    (*op).BindInputs({input_tensor1,input_tensor2}).BindOutputs(
        {output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Stack, LayoutinferernceTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 4, 6, 1});
  tim::vx::ShapeType kernel_shape({2, 2, 2, 3});
  tim::vx::ShapeType conv2dout_shape({3, 3, 5, 1});
  tim::vx::ShapeType output_shape({2, 5});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec conv2dout_spec(tim::vx::DataType::FLOAT32,
                                     conv2dout_shape,
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto conv2dout_tensor = graph->CreateTensor(conv2dout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1, 4, 2, 5, 3, 6, 3, 1, 1, 4, 2, 5, 3, 6, 3, 1, 1, 4, 2, 5, 3, 6, 3, 1,
      1, 4, 2, 5, 3, 6, 3, 1, 1, 4, 2, 5, 3, 6, 3, 1, 1, 4, 2, 5, 3, 6, 3, 1,
  };
  std::vector<float> kernel_data = {
      1, 4, 2, 5, 1, 2, 3, 6, 1, 4, 2, 5, 1, 2, 3, 6, 1, 4, 2, 5, 1, 2, 3, 6,
  };
  std::vector<float> golden = {
      1, 4, 2, 5, 3, 61, 1, 1, 1, 1,  //fake golden
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  // The following parameters have been reverse
  std::vector<int> begin = {0, 0, 0, 0};
  std::vector<int> end = {2, 3, 4, 1};
  std::vector<int> strides = {1, 1, 1, 1};
  uint32_t MASK_BEGIN = 0, MASK_END = 0b0110, MASK_SHRINK = 0b1010;

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(),
                                             in_data.size() * sizeof(float)));

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op2 = graph->CreateOperation<tim::vx::ops::Stack>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
