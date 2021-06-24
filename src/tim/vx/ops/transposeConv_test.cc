#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/tim/vx/test_utils.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/deconv.h"
#include "tim/vx/types.h"

TEST(TransposeConv2d, shape_4_4_1_1_float32_SimpleTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  // nchw
  std::vector<float> golden = {29,  62,  83,  75,  99,  192, 237, 198,
                               207, 372, 417, 330, 263, 446, 485, 365};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_4_4_2_1_float32_SameTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 2, 1});  //whio
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 2,  4,  6,  8,  10, 12,
                                   14, 16, 18, 20, 22, 24, 26, 28, 30, 32};

  // weight data   oihw
  std::vector<float> weight_data = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                                    2, 4, 6, 8, 10, 12, 14, 16, 18};

  // nchw
  std::vector<float> golden = {184,  412,  568,  528,  678,  1347, 1689, 1434,
                               1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_4_4_2_1_float32_ValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 2, 1});  //whio
  tim::vx::ShapeType output_shape({6, 6, 1, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 2,  4,  6,  8,  10, 12,
                                   14, 16, 18, 20, 22, 24, 26, 28, 30, 32};

  // weight data   oihw
  std::vector<float> weight_data = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                                    2, 4, 6, 8, 10, 12, 14, 16, 18};

  // nchw
  std::vector<float> golden = {
      5,   22,   59,   101,  114,  83,   52,  184,  412,  568,  528,  344,
      237, 678,  1347, 1689, 1434, 879,  597, 1494, 2715, 3057, 2442, 1431,
      856, 1968, 3352, 3652, 2760, 1548, 689, 1534, 2543, 2729, 2010, 1103};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_2_2_1_1_float32_StrideTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType output_shape({5, 5, 1, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  // nchw
  std::vector<float> golden = {1,  2,  5,  4,  6,  4,  5,  14, 10,
                               12, 10, 14, 36, 24, 30, 12, 15, 34,
                               20, 24, 21, 24, 55, 32, 36};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_2_2_1_1_float32_ChannelTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 2});  //whio
  tim::vx::ShapeType output_shape({5, 5, 2, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                                    2, 4, 6, 8, 10, 12, 14, 16, 18};

  // nchw
  std::vector<float> golden = {
      1,  3,  7,  6,  10, 7,   9,  25, 18, 22, 16, 24, 62, 42,  54, 21, 27,
      61, 36, 44, 39, 45, 103, 60, 68, 2,  4,  10, 8,  12, 8,   10, 28, 20,
      24, 20, 28, 72, 48, 60,  24, 30, 68, 40, 48, 42, 48, 110, 64, 72};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_2_1_1_1_float32_AccuracyTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 1, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType output_shape({4, 3, 1, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {323, 521};

  // weight data   oihw
  std::vector<float> weight_data = {9, 5, 6, 9, 8, 5, 3, 1, 4};

  // nchw
  std::vector<float> golden = {1615., 1938., 4689., 2605., 2584., 1615.,
                               4689., 4168., 323.,  1292., 1563., 521.};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({3, 3});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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

TEST(TransposeConv2d, shape_2_2_1_1_float32_BiasChannelTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape({5, 5, 2, 1});  //whcn

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::FLOAT32, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  // Input data  nchw
  std::vector<float> input_data = {1, 2, 3, 4};

  // weight data   oihw
  std::vector<float> weight_data = {1, 3, 5, 7, 9,  11, 13, 15, 17,
                                    2, 4, 6, 8, 10, 12, 14, 16, 18};

  // bias data
  std::vector<float> bias_data = {3, 4};

  // nchw
  std::vector<float> golden = {
      4,  6,  10, 9,  13, 10,  12, 28, 21, 25, 19, 27, 65, 45,  57, 24, 30,
      64, 39, 47, 42, 48, 106, 63, 71, 6,  8,  14, 12, 16, 12,  14, 32, 24,
      28, 24, 32, 76, 52, 64,  28, 34, 72, 44, 52, 46, 52, 114, 68, 76};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
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

TEST(TransposeConv2d, shape_4_4_1_1_uint8_QuantizedTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -508, OutputMax = 512;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  std::vector<float> input_data_float = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<u_int8_t> weight_data = {129, 131, 133, 135, 137,
                                       139, 141, 143, 145};

  // nchw
  std::vector<float> golden_float = {28,  64,  84,  76,  100, 192, 236, 200,
                                     208, 372, 416, 332, 264, 448, 484, 364};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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
  std::vector<float> output_float =
      Dequantize<uint8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_4_4_2_1_uint8_QuantizedTwoFiltersTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 2, 1});  //whio
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -4064, OutputMax = 4096;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  std::vector<float> input_data_float = {
      1, 3, 5, 7, 9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
      2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<u_int8_t> weight_data = {129, 133, 137, 141, 145, 149,
                                       153, 157, 161, 131, 135, 139,
                                       143, 147, 151, 155, 159, 163};

  // nchw
  std::vector<float> golden_float = {192,  416,  576,  544,  672,  1344,
                                     1696, 1440, 1504, 2720, 3072, 2432,
                                     1984, 3360, 3648, 2752};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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
  std::vector<float> output_float =
      Dequantize<uint8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_4_4_2_1_uint8_QuantizedValidTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 2, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 2, 1});  //whio
  tim::vx::ShapeType output_shape({6, 6, 1, 1});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -4064, OutputMax = 4096;

  std::pair<float, int32_t> scalesAndZp;

  scalesAndZp = QuantizationParams<u_int8_t>(InputMin, InputMax);
  std::vector<float> scalesInput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsInput = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(WeightMin, WeightMax);
  std::vector<float> scalesWeight = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsWeight = {scalesAndZp.second};

  scalesAndZp = QuantizationParams<u_int8_t>(OutputMin, OutputMax);
  std::vector<float> scalesOutput = {scalesAndZp.first};
  std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

  tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 2,
                                   scalesInput, zeroPointsInput);
  tim::vx::Quantization quantWeight(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesWeight, zeroPointsWeight);
  tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 2,
                                    scalesOutput, zeroPointsOutput);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
  tim::vx::TensorSpec weight_spec(tim::vx::DataType::UINT8, weight_shape,
                                  tim::vx::TensorAttribute::CONSTANT,
                                  quantWeight);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT,
                                  quantOutput);

  // Input data  nchw
  std::vector<float> input_data_float = {
      1, 3, 5, 7, 9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
      2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<u_int8_t> weight_data = {129, 133, 137, 141, 145, 149,
                                       153, 157, 161, 131, 135, 139,
                                       143, 147, 151, 155, 159, 163};

  // nchw
  std::vector<float> golden_float = {
      0,   32,   64,   96,   128,  96,   64,  192,  416,  576,  544,  352,
      224, 672,  1344, 1696, 1440, 864,  608, 1504, 2720, 3072, 2432, 1440,
      864, 1984, 3360, 3648, 2752, 1536, 704, 1536, 2528, 2720, 2016, 1088};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
      .BindInput(input_tensor)
      .BindInput(weight_tensor)
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
  std::vector<float> output_float =
      Dequantize<uint8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_4_4_1_1_uint8_QuantizedBiasTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  float InputMin = -63.5, InputMax = 64, WeightMin = -63.5, WeightMax = 64,
        OutputMin = -508, OutputMax = 512;

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
  std::vector<float> input_data_float = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<u_int8_t> input_data =
      Quantize<uint8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<u_int8_t> weight_data = {129, 131, 133, 135, 137,
                                       139, 141, 143, 145};
  // bias data
  std::vector<float> bias_data_float = {1};
  std::vector<int32_t> bias_data =
      Quantize<int32_t>(bias_data_float, scalesBias[0], zeroPointsBias[0]);

  // nchw
  std::vector<float> golden_float = {32,  64,  84,  76,  100, 192, 240, 200,
                                     208, 372, 420, 332, 264, 448, 488, 368};
  std::vector<u_int8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
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
  std::vector<float> output_float =
      Dequantize<uint8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_4_4_1_1_int8_QuantizedPerChannelOneTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  std::vector<float> scalesInput = {16.0 / 255};
  std::vector<int32_t> zeroPointsInput = {-128};

  std::vector<float> scalesWeight = {9.0 / 127};
  std::vector<int32_t> zeroPointsWeight = {0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  std::vector<float> scalesOutput = {2};
  std::vector<int32_t> zeroPointsOutput = {-128};

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
  std::vector<float> input_data_float = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<int8_t> weight_data = {14, 28, 42, 56, 71, 85, 99, 113, 127};
  // bias data
  std::vector<int32_t> bias_data = {0};
  // nchw
  std::vector<float> golden_float = {28,  62,  82,  76,  98,  192, 238, 198,
                                     206, 372, 416, 330, 262, 446, 486, 366};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
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
  std::vector<float> output_float =
      Dequantize<int8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_2_2_1_1_int8_QuantizedPerChannelTwoTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 2, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 2});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape({5, 5, 2, 1});  //whcn

  std::vector<float> scalesInput = {4.0 / 255};
  std::vector<int32_t> zeroPointsInput = {-128};

  std::vector<float> scalesWeight = {17.0 / 127, 18.0 / 127};
  std::vector<int32_t> zeroPointsWeight = {0, 0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  std::vector<float> scalesOutput = {1};
  std::vector<int32_t> zeroPointsOutput = {-128};

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
  std::vector<float> input_data_float = {1, 2, 3, 4};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<int8_t> weight_data = {7,  22, 37, 52, 67, 82, 97, 112, 127,
                                     14, 28, 42, 56, 71, 85, 99, 113, 127};
  // bias data
  std::vector<int32_t> bias_data = {0, 0};
  // nchw
  std::vector<float> golden_float = {
      1,  3,  7,  6,  10, 7,   9,  25, 18, 22, 16, 24, 62, 42,  54, 21, 27,
      61, 36, 44, 39, 45, 103, 60, 68, 2,  4,  10, 8,  12, 8,   10, 28, 20,
      24, 20, 28, 72, 48, 60,  24, 30, 68, 40, 48, 42, 48, 110, 64, 72};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::VALID;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
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
  std::vector<float> output_float =
      Dequantize<int8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}

TEST(TransposeConv2d, shape_4_4_1_1_int8_QuantizedBiasPerChannelTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 4, 1, 1});   //whcn
  tim::vx::ShapeType weight_shape({3, 3, 1, 1});  //whio
  tim::vx::ShapeType bias_shape({weight_shape[3]});
  tim::vx::ShapeType output_shape({4, 4, 1, 1});  //whcn

  std::vector<float> scalesInput = {16.0 / 255};
  std::vector<int32_t> zeroPointsInput = {-128};

  std::vector<float> scalesWeight = {9.0 / 127};
  std::vector<int32_t> zeroPointsWeight = {0};

  int32_t sizeofweight = scalesWeight.size();
  std::vector<float> scalesBias(sizeofweight);
  std::vector<int32_t> zeroPointsBias(sizeofweight);
  for (int i = 0; i < sizeofweight; i++) {
    scalesBias[i] = scalesInput[0] * scalesWeight[i];
    zeroPointsBias[i] = 0;
  }

  std::vector<float> scalesOutput = {2};
  std::vector<int32_t> zeroPointsOutput = {-128};

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
  std::vector<float> input_data_float = {1, 2,  3,  4,  5,  6,  7,  8,
                                         9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int8_t> input_data =
      Quantize<int8_t>(input_data_float, scalesInput[0], zeroPointsInput[0]);

  // weight data   oihw
  std::vector<int8_t> weight_data = {14, 28, 42, 56, 71, 85, 99, 113, 127};
  // bias data
  std::vector<int32_t> bias_data = {224};
  // nchw
  std::vector<float> golden_float = {30,  62,  84,  76,  100, 194, 238, 200,
                                     208, 372, 418, 330, 264, 446, 486, 366};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto output_tensor = graph->CreateTensor(output_spec);

  auto padding = tim::vx::PadType::SAME;
  std::array<uint32_t, 2> kernelSize({weight_shape[1], weight_shape[0]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> outputPadding({0, 0});
  int32_t pad_left_inter =
      static_cast<int32_t>(weight_shape[0] + stride[0] * (input_shape[0] - 1) -
                           output_shape[1]) / 2;
  uint32_t pad_left = pad_left_inter > 0 ? pad_left_inter : 0;
  uint32_t pad_right = pad_left;
  int32_t pad_top_inter =
      static_cast<int32_t>(weight_shape[1] + stride[1] * (input_shape[1] - 1) -
                           output_shape[0]) / 2;
  uint32_t pad_top = pad_top_inter > 0 ? pad_top_inter : 0;
  uint32_t pad_bottom = pad_top;
  std::array<uint32_t, 4> pad = {pad_left, pad_right, pad_top, pad_bottom};

  auto transposeConv2d = graph->CreateOperation<tim::vx::ops::DeConv2d>(
      weight_shape[3], padding, kernelSize, stride, outputPadding, pad);
  (*transposeConv2d)
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
  std::vector<float> output_float =
      Dequantize<int8_t>(output, scalesOutput[0], zeroPointsOutput[0]);
  EXPECT_THAT(output_float,
              ElementsAreArray(ArrayFloatNear(golden_float, scalesOutput[0])));
}
