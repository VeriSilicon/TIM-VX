#include "tim/vx/ops/conv2d.h"

#include "gtest/gtest.h"
#include "src/tim/vx/test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/types.h"

TEST(Conv2d, shape_4_2_1_1_float32_PaddingTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

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
      1, 1, 1, 1,  // row = 1
      2, 2, 3, 2   // row = 2
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1,  2,  3,  4,  //first 2x2 filter
      -1, 1,  -1, 1,  // second 2x2 filter
      -1, -1, 1,  1,  // third 2x2 filter
  };

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {// first channel
                               18, 22, 21, 8, 7, 9, 8, 3, 2, 3, 1, -1,
                               // second channel
                               2, 3, 1, 0, 5, 6, 6, 4, -1, -2, -2, 1};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_2_2_float32_PointwiseTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({1, 1, 2, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

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
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2  // first filter
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3,
                               1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_1_2_float32_SimpleTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

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
      // First batch
      1, 1, 1, 1,  // row = 1
      2, 2, 2, 2,  // row = 2
      // Second batch
      1, 2, 3, 4,  // row = 1
      1, 2, 3, 4,  // row = 2
  };

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_2_2_float32_SimpleChannelsTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data
  std::vector<float> input_data = {
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data
  std::vector<float> weight_data = {1,  2, 3,  4, 1,  2,  3, 4, -1, 1,  -1, 1,
                                    -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1,  1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  std::vector<float> golden = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_6_3_1_1_float32_SimpleAnisotropicStridesTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({6, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 2, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {3,  2,  1,  -1, -2, -3, 4,  3,  2,
                                   -2, -3, -4, 5,  4,  3,  -3, -4, -5};

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2,  //
      3, 4,  //
  };

  // bias data
  std::vector<float> bias_data = {-1};

  // nchw
  std::vector<float> golden = {
      30, -24,  //
      40, -34,  //
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({3, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {105, 150, 183, 95,  235, 312,
                               357, 178, 187, 234, 261, 121};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedConstFilterTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {105, 150, 183, 95,  235, 312,
                               357, 178, 187, 234, 261, 121};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedBiasTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 3, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {10};

  // nchw
  std::vector<float> golden = {115, 160, 193, 105, 245, 322,
                               367, 188, 197, 244, 271, 131};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_3_1_1_float32_HandCalculatedValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // weight data   oihw
  std::vector<float> weight_data = {1, 4, 7, 2, 5, 8, 3, 6, 9};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {312, 357};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_2_2_float32_DisabledPointwiseMultifilterTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});   //whcn
  tim::vx::ShapeType weight_shape({1, 1, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {4, 2, weight_shape[3], input_shape[3]});  //whcn

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
      0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1, 0.5, 0.5, 0.5, 0.5, 1,   1, 1,   1,
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 2, 3};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3, 2.5, 2.5, 2.5, 2.5, 5,   5, 5,   5,
      1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6, 2.5, 5,   7.5, 10,  2.5, 5, 7.5, 10};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_9_9_1_1_float32_SimpleDilationTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[3], input_shape[3]});  //whcn

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

  // weight data   oihw
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

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_1_2_float32_StrideTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 2, 2, 3, 2,
                                   1, 2, 3, 4, 1, 2, 4, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {18, 22, 21, 2, 3, 1, 5, 6, 6,
                               17, 31, 40, 4, 5, 3, 3, 4, 4};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_1_2_float32_InputAndFilterSameWidthHeightTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({4, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {1, 1, weight_shape[3], input_shape[3]});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 1, 1, 1, 2, 2, 2, 2,
                                   1, 2, 3, 4, 1, 2, 3, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, -1, -1, 1, 1};

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {10, 34};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_4_2_1_2_uint8_QuantizedTest1) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

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
  std::vector<float> input_data_float = {1, 1, 1, 1, 2, 2, 2, 2,
                                         1, 2, 3, 4, 1, 2, 3, 4};
  // weight data   oihw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> weight_data_float = {1,  2, 3,  4,  -1, 1,
                                          -1, 1, -1, -1, 1,  1};
  // bias data
  // scale:0.25  Zp:0
  std::vector<float> bias_data_float = {1, 2, 3};
  // golden data
  //min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);
  std::vector<u_int8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);
  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_4_2_1_2_uint8_QuantizedTest2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({4, 2, 1, 2});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 3});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float InputMin = -128.5, InputMax = 128, WeightMin = -128.5, WeightMax = 128,
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
  // min:-128.5  max:128  scale:1.00588  Zp:0
  std::vector<float> input_data_float = {1, 1, 1, 1, 2, 2, 2, 2,
                                         1, 2, 3, 4, 1, 2, 3, 4};
  // weight data   oihw
  // min:-128.5  max:128  scale:1.00588  Zp:0
  std::vector<float> weight_data_float = {1,  2, 3,  4,  -1, 1,
                                          -1, 1, -1, -1, 1,  1};
  // bias data
  // scale:1.0116  Zp:0
  std::vector<float> bias_data_float = {1, 2, 3};
  // golden data
  // min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);
  std::vector<u_int8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_6_3_1_1_uint8_AnisotropicStridesQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({6, 3, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 2, weight_shape[3], input_shape[3]});  //whcn

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
  std::vector<float> input_data_float = {3,  2,  1,  -1, -2, -3, 4,  3,  2,
                                         -2, -3, -4, 5,  4,  3,  -3, -4, -5};
  // weight data   oihw
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> weight_data_float = {1, 2, 3, 4};
  // bias data
  // scale:0.25  Zp:0
  std::vector<float> bias_data_float = {-1};
  // golden data
  //min:-127  max:128  scale:1  Zp:-1
  std::vector<float> golden_float = {30, -24, 40, -34};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);
  std::vector<u_int8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({3, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_9_9_1_1_uint8_DilationQuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({9, 9, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {3, 3, weight_shape[3], input_shape[3]});  //whcn

  float InputMin = -128, InputMax = 127, WeightMin = -128, WeightMax = 127,
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
  // min:-128  max:127  scale:1  Zp:0
  std::vector<float> input_data_float = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
      0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // weight data   oihw
  // min:-128  max:127  scale:1  Zp:0
  std::vector<float> weight_data_float = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  // bias data
  // scale:1  Zp:0
  std::vector<float> bias_data_float = {0};
  // golden data
  // min:0  max:255  scale:1  Zp:-128
  std::vector<float> golden_float = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);
  std::vector<u_int8_t> weight_data =
      Quantize<uint8_t>(weight_data_float, scalesWeight[0], zeroPointsInput[0]);
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({3, 3});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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
  std::vector<u_int8_t> output(output_size);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(Conv2d, shape_3_2_2_1_int8_QuantizedPerTensorTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {1};
  std::vector<int32_t> zeroPointsWeight = {0};

  std::vector<float> scalesBias = {scalesInput[0] * scalesWeight[0]};
  std::vector<int32_t> zeroPointsBias = {0};

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
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight_float_data = {1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2};
  std::vector<int8_t> weight_data = {1, 3, 3, 5, 2, 4, 4, 6,
                                     7, 5, 3, 1, 8, 6, 4, 2};

  // bias data
  std::vector<float> bias_data_float = {3, -2};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);

  // golden_int8_data = {61, -115, 111, -89}
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> golden_float = {31, -57, 56, -44};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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

TEST(Conv2d, shape_3_2_2_1_int8_QuantizedPerChannelTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({3, 2, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({2, 2, 2, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape(
      {2, 1, weight_shape[3], input_shape[3]});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = 0, WeightMax = 0,
        OutputMin = -63.5, OutputMax = 64;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {1, 2};
  std::vector<int32_t> zeroPointsWeight = {0, 0};

  std::vector<float> scalesBias = {scalesInput[0] * scalesWeight[0],
                                   scalesInput[0] * scalesWeight[1]};
  std::vector<int32_t> zeroPointsBias = {0, 0};

  scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL,
                                    3, scalesWeight, zeroPointsWeight);
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
  // min:-63.5   max:64   scale:0.5  Zp:-1
  std::vector<float> input_data_float = {3, 1,  -2, 4, 2,  -3,
                                         2, -1, -3, 3, -2, -4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight_data_float = {1, 3, 3, 5, 2, 4, 4, 6, 7, 5, 3, 1, 8, 6, 4, 2};
  std::vector<int8_t> weight_data = {1, 3, 3, 5, 2, 4, 4, 6,
                                     4, 3, 2, 1, 4, 3, 2, 1};

  // bias_data_float ={3, -2};
  std::vector<int32_t> bias_data = {6, -2};

  // golden data
  // min:-63.5  max:64  scale:0.5  Zp:-1
  std::vector<float> golden_float = {31, -57, 64, -46};
  std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({1, 1});

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      padding, stride, dilation);
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
