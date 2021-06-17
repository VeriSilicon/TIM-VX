#include "gtest/gtest.h"
#include "src/tim/vx/test_utils.h"
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

TEST(DepthwiseConv, shape_2_3_2_1_float32_StrideValidTest) {
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

TEST(DepthwiseConv, shape_2_3_2_1_float32_StrideSameTest) {
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
  std::vector<float> golden = {71, -93, -34, 122, 99, -111, -20, 172};

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

TEST(DepthwiseConv, shape_2_3_2_1_float32_StrideSameDilationTest) {
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
  std::vector<float> golden = {1, 1, 2, 2, 3, 3, 4, 4};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({10, 10});
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

TEST(DepthwiseConv, shape_2_3_2_1_uint8_QuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 2, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -127, OutputMax = 128;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  std::vector<float> scalesBias = {scalesInput[0] * scalesWeight[0]};
  std::vector<int32_t> zeroPointsBias = {0};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::ASYMMETRIC, 2, scalesBias,
                                  zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12};
  std::vector<uint8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> weight_data_float = {1, -9,  5, 13, 2, 10, 6, -14,
                                          3, -11, 7, 15, 4, 12, 8, -16};
  std::vector<uint8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);

  // bias data
  // scale:0.25  Zp:0
  std::vector<float> bias_data_float = {1, 2, 3, 4};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);

  // golden
  // min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {71, 91, -34, -26, 99, 127, -20, -4};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<uint8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_9_9_1_1_uint8_QuantizedDilationdValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = 0, InputMax = 255, WeightMin = 0, WeightMax = 255,
        OutputMin = 0, OutputMax = 255;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  std::vector<float> scalesBias = {scalesInput[0] * scalesWeight[0]};
  std::vector<int32_t> zeroPointsBias = {0};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::ASYMMETRIC, 2, scalesBias,
                                  zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> input_data_float = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> weight_data_float = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<uint8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);

  // bias data
  // scale:1  Zp:0
  std::vector<float> bias_data_float = {0};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);

  // golden
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> golden_float = {5, 5, 5, 5, 5, 5, 5, 5, 5};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<uint8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_3_1_1_uint8_QuantizedDilationdSameTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = 0, InputMax = 255, WeightMin = 0, WeightMax = 255,
        OutputMin = 0, OutputMax = 255;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  std::vector<float> scalesBias = {scalesInput[0] * scalesWeight[0]};
  std::vector<int32_t> zeroPointsBias = {0};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::ASYMMETRIC, 2, scalesBias,
                                  zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> input_data_float = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> weight_data_float = {1, 2, 3, 4};
  std::vector<uint8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);

  // bias data
  // scale:1  Zp:0
  std::vector<float> bias_data_float = {0};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);

  // golden
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> golden_float = {4, 7, 3, 6, 10, 4, 2, 3, 1};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<uint8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_2_2_1_int8_PerTensorTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  std::vector<float> scalesWeight = {1};
  std::vector<int32_t> zeroPointsWeight = {0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::ASYMMETRIC, 2, scalesBias,
                                  zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  std::vector<float> weight_data_float = {1, 3, 7, 3, 2, 4, 8, 4,
                                          3, 5, 5, 1, 4, 6, 6, 2};
  std::vector<int8_t> weight_data =
      Quantize<int8_t>(weight_data_float, scalesWeight[0], zeroPointsWeight[0]);

  // bias data
  std::vector<int32_t> bias_data = {6, -4, 8, 12};

  // golden
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {43, 3, 48, -4, 18, -28, 22, -36};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_2_2_1_int8_PerAxisTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 4, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  std::vector<float> scalesWeight = {1, 2, 3, 4};
  std::vector<int32_t> zeroPointsWeight = {0, 0, 0, 0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                    2, scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                  scalesBias, zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};

  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  std::vector<int8_t> weight_data = {1, 3, 7, 3, 1, 2, 4, 2,
                                     1, 2, 2, 0, 1, 2, 2, 1};

  // bias data
  std::vector<int32_t> bias_data = {6, -2, 2, 3};

  // golden
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {43, 3, 48, -4, 21, -30, 22, -54};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_3_8_1_int8_PerChannelValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 8, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 8, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  std::vector<float> scalesWeight = {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1};
  std::vector<int32_t> zeroPointsWeight = {0, 0, 0, 0, 0, 0, 0, 0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                    2, scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                  scalesBias, zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  std::vector<int8_t> weight_data = {
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 20, 20, 20, 20, 20, 20, 20, 20, 20,
      35, 35, 35, 35, 35, 35, 35, 35, 35, 80, 80, 80, 80, 80, 80, 80, 80, 80};

  // bias data
  std::vector<int32_t> bias_data = {0, 0, 0, 0, 0, 0, 0, 0};

  // golden
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {9, 18, 0, 0, 47, 54, 0, 0};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(DepthwiseConv, shape_3_3_8_1_int8_PerChannelSameTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({3, 3, 8, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 8, 1});  //whoi
  tim::vx::ShapeType bias_shape({weight_shape[2]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[2], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  std::vector<float> scalesWeight = {0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1};
  std::vector<int32_t> zeroPointsWeight = {0, 0, 0, 0, 0, 0, 0, 0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};
  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                    2, scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantBias(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, 0,
                                  scalesBias, zeroPointsBias);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::INT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::INT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT, quantBias);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> input_data_float = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   iohw
  std::vector<int8_t> weight_data = {
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 20, 20, 20, 20, 20, 20, 20, 20, 20,
      35, 35, 35, 35, 35, 35, 35, 35, 35, 80, 80, 80, 80, 80, 80, 80, 80, 80};

  // bias data
  std::vector<int32_t> bias_data = {0, 0, 0, 0, 0, 0, 0, 0};

  // golden
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {
      4,  6,  4,  6,  9,  6,  4,  6,  4,  8,  12, 8,  12, 18, 12, 8,  12, 8,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      21, 31, 21, 31, 47, 31, 21, 31, 21, 24, 36, 24, 36, 54, 36, 24, 36, 24,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

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
  std::vector<int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}
