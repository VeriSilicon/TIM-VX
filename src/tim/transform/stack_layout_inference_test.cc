#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"

TEST(Stack, DISABLED_LayoutinferernceTest_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 4, 1});      //cwhn
  tim::vx::ShapeType kernel_shape({2, 3, 3, 3});     //iwho
//   tim::vx::ShapeType conv2dout_shape({3, 1, 2, 1});  //cwhn
  tim::vx::ShapeType output_shape({2, 3, 1, 2, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec conv2dout_spec(tim::vx::DataType::FLOAT32,
                                     {0,0,0,0},
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto conv2dout_tensor = graph->CreateTensor(conv2dout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1, 1, 1, 1, 2, 0, 5, 3, 6, 3, 1, 1,
      1, 4, 2, 5, 7, 6, 3, 1, 1, 0, 2, 5,
  };
  std::vector<float> kernel_data = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2, 1, 1, 1,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1,
      1, 1, 1, 1, 2, 1, 1, 5, 3, 1, 2, 3, 1, 1, 2, 1, 1, 1,
  };
  std::vector<float> golden = {
      64, 64, 49, 49, 81, 81, 77, 77, 44, 44, 97, 97
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});

  auto op2 = graph->CreateOperation<tim::vx::ops::Stack>(0, 2);
  (*op2).BindInputs({conv2dout_tensor, conv2dout_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Stack, LayoutinferernceTest_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 4, 1});      //cwhn
  tim::vx::ShapeType kernel_shape({2, 2, 3, 3});     //iwho
  tim::vx::ShapeType conv2dout_shape({3, 2, 2, 1});  //cwhn
//   tim::vx::ShapeType output_shape({2, 1, 2, 1});
  tim::vx::ShapeType output_shape({2, 3, 2});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec conv2dout_spec(tim::vx::DataType::FLOAT32,
                                     conv2dout_shape,
                                     tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec reduceout_spec(tim::vx::DataType::FLOAT32,
                                     {0,0,0},
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto conv2dout_tensor = graph->CreateTensor(conv2dout_spec);
  auto reduceout_tensor = graph->CreateTensor(reduceout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1, 1, 1, 1, 2, 0, 5, 3, 6, 3, 1, 1,
      1, 4, 2, 5, 7, 6, 3, 1, 1, 0, 2, 5,
  };
  std::vector<float> kernel_data = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  std::vector<float> golden = {
      33, 33, 37, 35, 35, 43, 34, 34, 58, 39, 39, 43,
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  std::vector<int32_t> axis = {2,3};
  auto op2 = graph->CreateOperation<tim::vx::ops::ReduceMax>(axis, false);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({reduceout_tensor});
  auto op3 = graph->CreateOperation<tim::vx::ops::Stack>(0, 2);
  (*op3).BindInputs({reduceout_tensor, reduceout_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Stack, LayoutinferernceTest_3) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 4, 1});      //cwhn
  tim::vx::ShapeType kernel_shape({2, 2, 3, 3});     //iwho
  tim::vx::ShapeType conv2dout_shape({3, 2, 2, 1});  //cwhn
  tim::vx::ShapeType output_shape({2, 3, 2, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec conv2dout_spec(tim::vx::DataType::FLOAT32,
                                     {0,0,0,0},
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec reduceout_spec(tim::vx::DataType::FLOAT32,
                                     {0,0,0},
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto input2_tensor = graph->CreateTensor(input_spec);
  auto conv2dout_tensor = graph->CreateTensor(conv2dout_spec);
  auto conv2dout2_tensor = graph->CreateTensor(conv2dout_spec);
  auto reduceout_tensor = graph->CreateTensor(reduceout_spec);
  auto reduceout2_tensor = graph->CreateTensor(reduceout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      1, 1, 1, 1, 2, 0, 5, 3, 6, 3, 1, 1,
      1, 4, 2, 5, 7, 6, 3, 1, 1, 0, 2, 5,
  };
  std::vector<float> kernel_data = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 2, 1, 1, 1,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1,
      1, 1, 1, 1, 2, 1, 1, 5, 3, 1, 2, 3, 1, 1, 2, 1, 1, 1,
  };
  std::vector<float> golden = {
      55, 49, 21, 28, 37, 40, 39, 55, 28, 24, 41, 41,
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());
  auto kernel2_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});
  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op11 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op11)
      .BindInputs({input2_tensor, kernel2_tensor})
      .BindOutputs({conv2dout2_tensor});

  std::vector<int32_t> axis = {1};
  auto op2 = graph->CreateOperation<tim::vx::ops::ReduceMax>(axis, false);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({reduceout_tensor});
  axis = {2};
  auto op22 = graph->CreateOperation<tim::vx::ops::ReduceMax>(axis, false);
  (*op22).BindInputs({conv2dout2_tensor}).BindOutputs({reduceout2_tensor});

  auto op3 = graph->CreateOperation<tim::vx::ops::Stack>(0, 2);
  (*op3).BindInputs({reduceout_tensor, reduceout2_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_input2 = graph_io_map[graph->InputsTensor()[1]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_input2->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
