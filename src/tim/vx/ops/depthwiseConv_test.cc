#include "gtest/gtest.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/types.h"

TEST(DepthwiseConv, shape_2_3_2_1_float32_SimpleTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 2, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12};

  // weight data   iohw
  std::vector<float> weight_data = {1, -9,  5, 13, 2, 10, 6, -14,
                                    3, -11, 7, 15, 4, 12, 8, -16};

  // bias data
  std::vector<float> bias_data = {1, 2, 3, 4};

  // nchw
  std::vector<float> golden = {71, 91, -34, -26, 99, 127, -20, -4};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_2_3_2_1_float32_StrideTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12};

  // weight data   iohw
  std::vector<float> weight_data = {1, -9,  5, 13, 2, 10, 6, -14,
                                    3, -11, 7, 15, 4, 12, 8, -16};

  // bias data
  std::vector<float> bias_data = {1, 2, 3, 4};

  // nchw
  std::vector<float> golden = {71, -34, 99, -20};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_2_3_2_1_float32_PaddingTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12};

  // weight data   iohw
  std::vector<float> weight_data = {1, -9,  5, 13, 2, 10, 6, -14,
                                    3, -11, 7, 15, 4, 12, 8, -16};

  // bias data
  std::vector<float> bias_data = {1, 2, 3, 4};

  // nchw
  std::vector<float> golden = {71, -34, 99, -20};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_9_9_1_1_float32_DilationValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // weight data   iohw
  std::vector<float> weight_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({3, 3});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_3_1_1_float32_DilationSameTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  // weight data   iohw
  std::vector<float> weight_data = {1, 2, 3, 4};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {4, 7, 3, 6, 10, 4, 2, 3, 1};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({2, 2});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_3_4_2_float32_BatchValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 4, 2});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // weight data   iohw
  std::vector<float> weight_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
                                    3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4};

  // bias data
  std::vector<float> bias_data = {0, 0, 0, 0};

  // nchw
  std::vector<float> golden = {9, 18, 0, 0, 9, 18, 0, 0};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_2_2_1_4_float32_BatchSameTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 1, 4});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {2, 2, weight_shape[2], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 0, 0, 0, 0,
                                   1, 1, 2, 2, 2, 2, 2, 2};

  // weight data   iohw
  std::vector<float> weight_data = {1, 1, 1, 0, 2, 0, 1, 1, 1};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 6, 8, 8, 8, 8};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});
  int32_t multiplier = weight_shape[2] / input_shape[2];

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation, multiplier);
  (*conv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
      .BindInput(bias_tensor)
      .BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());

  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());

  uint32_t output_size = 1;
  for (auto i : output_tensor->GetShape()) {
    output_size *= i;
  }
  std::vector<float> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
