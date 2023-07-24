#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"

#include <algorithm>

#include "gtest/gtest.h"

TEST(StridedSlice, endmask_2_shrinkmask_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 4, 6, 1});
  tim::vx::ShapeType kernel_shape({3, 2, 2, 2});
  tim::vx::ShapeType conv2dout_shape({3, 3, 5, 1});
  tim::vx::ShapeType output_shape({2, 3, 1});

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
      1, 0, 3, 4, 4, 2, 1, 2, 3, 1, 3, 1, 1, 3, 1, 0, 2, 0, 3, 1, 4, 0, 0, 2,
  };
  std::vector<float> golden = {
      55, 30, 71, 40, 40, 38,
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  // The following parameters have been reverse
  std::vector<int> begin = {0, 0, 0, 0};
  std::vector<int> end = {2, 3, 4, 1};
  std::vector<int> strides = {1, 1, 1, 1};
  uint32_t MASK_BEGIN = 0, MASK_END = 0b0100, MASK_SHRINK = 0b0100;

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op2 = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

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

TEST(StridedSlice, endmask_6_shrinkmask_5) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 4, 6, 1});
  tim::vx::ShapeType kernel_shape({3, 2, 2, 2});
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
      1, 0, 3, 4, 4, 2, 1, 2, 3, 1, 3, 1, 1, 3, 1, 0, 2, 0, 3, 1, 4, 0, 0, 2,
  };
  std::vector<float> golden = {55, 30, 55, 30, 55, 30, 55, 30, 55, 30};
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  // The following parameters have been reverse
  std::vector<int> begin = {0, 0, 0, 0};
  std::vector<int> end = {2, 3, 4, 1};
  std::vector<int> strides = {1, 1, 1, 1};
  uint32_t MASK_BEGIN = 0, MASK_END = 0b0110, MASK_SHRINK = 0b1010;

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op2 = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

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

TEST(StridedSlice, endmask_1_shrinkmask_1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 4, 6, 1});   //tf layout
  tim::vx::ShapeType kernel_shape({2, 2, 2, 3});  //tf layout
  tim::vx::ShapeType conv2dout_shape({3, 3, 5, 1});
  tim::vx::ShapeType output_shape({2, 3, 4});

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
      1, 0, 3, 4, 4, 2, 1, 2, 3, 1, 3, 1, 1, 3, 1, 0, 2, 0, 3, 1, 4, 0, 0, 2,
  };
  std::vector<float> golden = {
      51, 33, 68, 46, 45, 49, 51, 33, 68, 46, 45, 49,
      51, 33, 68, 46, 45, 49, 51, 33, 68, 46, 45, 49,
  };
  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  // The following parameters have been reverse
  std::vector<int> begin = {0, 0, 0, 0};
  std::vector<int> end = {2, 3, 4, 1};
  std::vector<int> strides = {1, 1, 1, 1};
  uint32_t MASK_BEGIN = 0, MASK_END = 0b1000, MASK_SHRINK = 0b1000;

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op2 = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

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

TEST(StridedSlice, beginmask_9_endmask_15) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({44, 58, 58, 1});   //tflite layout, cwhn
  tim::vx::ShapeType kernel_shape({44,2,2,44});  //tflite layout, iwho
//   tim::vx::ShapeType conv2dout_shape({44, 57, 57, 1}); //cwhn
  tim::vx::ShapeType output_shape({44, 56, 56, 1});

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

  std::vector<float> in_data;
  for (uint32_t i = 0; i < 44*58*58; ++i) {
    in_data.push_back(0.5);
  };
  std::vector<float> kernel_data;
  for (uint32_t i = 0; i < 44*4*44; ++i) {
    kernel_data.push_back(0.5);
  };

  auto kernel_tensor = graph->CreateTensor(kernel_spec, kernel_data.data());

  // The following parameters have been reverse
  std::vector<int> begin = {0, 1, 1, 0};
  std::vector<int> end = {0, 0, 0, 0};
  std::vector<int> strides = {1, 1, 1, 1};
  uint32_t MASK_BEGIN = 0b1001, MASK_END = 0b1111, MASK_SHRINK = 0b0000;

  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto op1 = graph->CreateOperation<tim::vx::ops::Conv2d>(
      tim::vx::PadType::VALID, stride, dilation, 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*op1)
      .BindInputs({input_tensor, kernel_tensor})
      .BindOutputs({conv2dout_tensor});
  auto op2 = graph->CreateOperation<tim::vx::ops::StridedSlice>(
      begin, end, strides, MASK_BEGIN, MASK_END, MASK_SHRINK);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(44*56*56);
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
}
