#include "tim/vx/ops/conv2d.h"

#include "tim/transform/layout_inference.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"

#include "gtest/gtest.h"

TEST(OP, conv2d_PaddingTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({2, 2, 1, 3}); //whio  
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

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_PointwiseFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});  //whcn 
  tim::vx::ShapeType weight_shape({1, 1, 2, 1}); //whio  
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
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2
      };

  // weight data   oihw
  std::vector<float> weight_data = {
    1, 2  // first filter
    };

    // bias data
    std::vector<float> bias_data = {0};

    // nchw
    std::vector<float> golden = {
        1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3,                         
        1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6
        };

    auto input_tensor = graph->CreateTensor(input_spec);
    auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
    auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

    auto output_tensor = graph->CreateTensor(output_spec);

    std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
    std::array<uint32_t, 2> stride({1, 1});
    std::array<uint32_t, 2> dilation({0, 0});
    auto padding = tim::vx::PadType::SAME;

    auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
        weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_SimpleTestFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});  //whcn 
  tim::vx::ShapeType weight_shape({2, 2, 1, 3}); //whio  
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
  std::vector<float> weight_data = {
      1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1
  };

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {18, 18, 2, 2, 5, 5, 17, 37, 4, 4, 3, 3};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_SimpleTestFloat32WithChannels) {
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

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({2, 2});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_SimpleTestFloat32WithAnisotropicStrides) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({6, 3, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({2, 2, 1, 1}); //whio  
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
  std::vector<float> input_data = {
      3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4, 5, 4, 3, -3, -4, -5
  };

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

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({3, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_HandCalculatedFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({3, 3, 1, 1}); //whio  
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
  std::vector<float> input_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 4, 7, 2, 5, 8, 3, 6, 9
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      105, 150, 183, 95,  235, 312,
      357, 178, 187, 234, 261, 121
    };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_HandCalculatedFloat32WithConstFilter) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({3, 3, 1, 1}); //whio  
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
  std::vector<float> input_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 4, 7, 2, 5, 8, 3, 6, 9
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      105, 150, 183, 95,  235, 312,
      357, 178, 187, 234, 261, 121
    };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_HandCalculatedWithBiasFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({3, 3, 1, 1}); //whio  
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
  std::vector<float> input_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 4, 7, 2, 5, 8, 3, 6, 9
  };

  // bias data
  std::vector<float> bias_data = {10};

  // nchw
  std::vector<float> golden = {
      115, 160, 193, 105, 245, 322, 367, 188, 197, 244, 271, 131
    };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::SAME;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_HandCalculatedValidFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 3, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({3, 3, 1, 1}); //whio  
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
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 4, 7, 2, 5, 8, 3, 6, 9
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      312, 357
    };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_DISABLED_PointwiseMultifilterFloat32) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 2, 2});  //whcn 
  tim::vx::ShapeType weight_shape({1, 1, 2, 2}); //whio  
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
      0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2, 0.5, 1,   1.5, 2,   0.5, 1, 1.5, 2
    };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2, 2, 3
    };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      1.5, 1.5, 1.5, 1.5, 3,   3, 3,   3, 2.5, 2.5, 2.5, 2.5, 5,   5, 5,   5,
      1.5, 3,   4.5, 6,   1.5, 3, 4.5, 6, 2.5, 5,   7.5, 10,  2.5, 5, 7.5, 10
    };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_SimpleTestFloatWithDilation) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({9, 9, 1, 1});  //whcn 
  tim::vx::ShapeType weight_shape({3, 3, 1, 1}); //whio  
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
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2, 3, 4, 5, 6, 7, 8, 9
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({3, 3});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_StrideTest) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});  //whcn 
  tim::vx::ShapeType weight_shape({2, 2, 1, 3}); //whio  
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
  std::vector<float> input_data = {
      1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 3, 4, 1, 2, 4, 4
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1
  };

  // bias data
  std::vector<float> bias_data = {1, 2, 3};

  // nchw
  std::vector<float> golden = {
      18, 22, 21, 2, 3, 1, 5, 6, 6, 17, 31, 40, 4, 5, 3, 3, 4, 4
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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

TEST(OP, conv2d_InputAndFilterSameWidthHeight) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({4, 2, 1, 2});  //whcn 
  tim::vx::ShapeType weight_shape({4, 2, 1, 1}); //whio  
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
  std::vector<float> input_data = {
      1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2, 3, 4
  };

  // weight data   oihw
  std::vector<float> weight_data = {
      1, 2, 3, 4, -1, -1, 1, 1
  };

  // bias data
  std::vector<float> bias_data = {0};

  // nchw
  std::vector<float> golden = {
      10, 34
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weight_tensor = graph->CreateTensor(weight_spec, weight_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());

  auto output_tensor = graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> ksize({weight_shape[1], weight_shape[2]});
  std::array<uint32_t, 2> stride({1, 1});
  std::array<uint32_t, 2> dilation({0, 0});
  auto padding = tim::vx::PadType::VALID;

  auto conv2d = graph->CreateOperation<tim::vx::ops::Conv2d>(
      weight_shape[3], padding, ksize, stride, dilation);
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


