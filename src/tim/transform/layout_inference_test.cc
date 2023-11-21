#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(LayoutInference, simple_conv2d) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({1, 3, 3, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  auto input = src_graph->CreateTensor(input_spec);

  tim::vx::ShapeType kernel_shape({1, 2, 2, 1});
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  std::vector<float> kernel_data = {
      0.25f, 0.25f, 0.25f, 0.25f};
  auto kernel = src_graph->CreateTensor(kernel_spec, kernel_data.data());

  tim::vx::ShapeType bias_shape({1});
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  std::vector<float> bias_data = {0.0f};
  auto bias = src_graph->CreateTensor(bias_spec, bias_data.data());

  tim::vx::ShapeType output_shape({1, 2, 2, 1});
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output = src_graph->CreateTensor(output_spec);

  auto conv2d = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
      kernel_shape[0], tim::vx::PadType::AUTO,
      std::array<uint32_t, 2>({kernel_shape[2], kernel_shape[1]}),
      std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
      std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN,
      tim::vx::DataLayout::IcWHOc);
  (*conv2d).BindInputs({input, kernel, bias}).BindOutput(output);
  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f};
  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(float));
  infer_graph->Run();
  std::vector<float> out_data;
  auto infer_out_shape = infer_output->GetShape();
  out_data.resize(infer_out_shape[0] * infer_out_shape[1] * infer_out_shape[2] *
                  infer_out_shape[3]);
  infer_output->CopyDataFromTensor(out_data.data());
  std::vector<float> expect_output = {0.875f, 0.875f, 0.875f, 0.875f};
  EXPECT_TRUE(0 == memcmp((void*)out_data.data(), (void*)expect_output.data(),
                          sizeof(float) * out_data.size()));
  tim::vx::ShapeType expect_shape({1, 2, 2, 1});
  EXPECT_EQ(infer_out_shape, expect_shape);
}

TEST(LayoutInference, weight_as_input_conv2d) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();
  tim::vx::ShapeType input_shape({3, 3, 1, 1}); //WHCN
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  auto input = src_graph->CreateTensor(input_spec);

  tim::vx::ShapeType kernel_shape({1, 2, 2, 1}); //IWHO
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::INPUT);
  auto kernel = src_graph->CreateTensor(kernel_spec);

  tim::vx::ShapeType bias_shape({1});
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::INPUT);
  auto bias = src_graph->CreateTensor(bias_spec);

  tim::vx::ShapeType output_shape({2, 2, 1, 1}); //WHCN
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto output = src_graph->CreateTensor(output_spec);

  auto conv2d = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
      0, tim::vx::PadType::AUTO, std::array<uint32_t, 2>({0, 0}),
      std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({1, 1}),
      std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::WHCN,
      tim::vx::DataLayout::IcWHOc);
  (*conv2d).BindInputs({input, kernel, bias}).BindOutput(output);
  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();
  std::vector<float> input_data = {1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> kernel_data = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<float> bias_data = {0.0f};
  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_weight = graph_io_map[src_graph->InputsTensor()[1]];
  auto infer_bias = graph_io_map[src_graph->InputsTensor()[2]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(float));
  infer_weight->CopyDataToTensor(kernel_data.data(), kernel_data.size() * sizeof(float));
  infer_bias->CopyDataToTensor(bias_data.data(), bias_data.size() * sizeof(float));
  infer_graph->Run();
  std::vector<float> out_data;
  auto infer_out_shape = infer_output->GetShape();
  out_data.resize(infer_out_shape[0] * infer_out_shape[1] * infer_out_shape[2] *
                  infer_out_shape[3]);
  infer_output->CopyDataFromTensor(out_data.data());
  std::vector<float> expect_output = {0.875f, 0.875f, 0.875f, 0.875f};
  EXPECT_TRUE(0 == memcmp((void*)out_data.data(), (void*)expect_output.data(),
                          sizeof(float) * out_data.size()));
  tim::vx::ShapeType expect_shape({2, 2, 1, 1});
  EXPECT_EQ(infer_out_shape, expect_shape);
}

TEST(GroupedConv2d, kernel_bigger_than_input_SAME) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 1});  //whcn
  tim::vx::ShapeType kernel_shape({1, 3, 2, 2});  //iwho, i*groups=c
  tim::vx::ShapeType bias_shape({2});
  tim::vx::ShapeType output_shape({2, 3, 2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> in_data = {1.0f, 3.0f, 4.0f, 2.0f, 2.0f, 3.0f,
                                2.0f, 4.0f, 3.0f, 1.0f, 3.0f, 3.0f};
  std::vector<float> weight = {100.0f, 20.0f, 1.0f, 200.0f, 10.0f, 2.0f,
                               200.0f, 30.0f, 1.0f, 100.0f, 20.0f, 3.0f};
  std::vector<float> bias = {500.0f, -1000.0f};
  std::vector<float> golden = {567.0f,  1480.0f, 608.0f,  1370.0f,
                               543.0f,  760.0f,  -873.0f, -160.0f,
                               -840.0f, -10.0f,  -907.0f, -310.0f};
  auto input_tensor = src_graph->CreateTensor(input_spec);
  auto weight_tensor = src_graph->CreateTensor(kernel_spec, weight.data());
  auto bias_tensor = src_graph->CreateTensor(bias_spec, bias.data());
  auto output_tensor = src_graph->CreateTensor(output_spec);

  std::array<uint32_t, 2> dilations = {0, 0};
  std::array<uint32_t, 2> strides = {1, 1};
  auto op = src_graph->CreateOperation<tim::vx::ops::GroupedConv2d>(
      tim::vx::PadType::SAME, strides, dilations, 2, tim::vx::DataLayout::WHCN,
      tim::vx::DataLayout::IcWHOc);
  (*op).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});

  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();

  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_graph->Run();

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(FC, share_const_tensor) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 1});
  tim::vx::ShapeType kernel_shape({2, 2});
  tim::vx::ShapeType bias_shape({2});
  tim::vx::ShapeType output_shape({2, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec tran_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  std::vector<float> in_data = {1,4,};
  std::vector<float> weight = {-3,3,2,1,};
  std::vector<float> bias = {0.1, 0.4,};
  std::vector<float> golden = {-8, 25};
  auto input_tensor = src_graph->CreateTensor(input_spec);
  auto weight_tensor = src_graph->CreateTensor(kernel_spec, weight.data());
  auto bias_tensor = src_graph->CreateTensor(bias_spec, bias.data());
  auto tran_tensor = src_graph->CreateTensor(tran_spec);
  auto output_tensor = src_graph->CreateTensor(output_spec);

  auto op1 = src_graph->CreateOperation<tim::vx::ops::FullyConnected>(0,2);
  (*op1).BindInputs({input_tensor, weight_tensor, bias_tensor}).BindOutputs({tran_tensor});

  auto op2 = src_graph->CreateOperation<tim::vx::ops::FullyConnected>(0,2);
  (*op2).BindInputs({tran_tensor, weight_tensor, bias_tensor}).BindOutputs({output_tensor});
  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();

  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_graph->Run();

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}

TEST(InstanceNorm, nhwc) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({2, 2, 2, 2}); //nhwc
  tim::vx::ShapeType param_shape({2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec param_spec(tim::vx::DataType::FLOAT32,
                            param_shape, tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = src_graph->CreateTensor(input_spec);
  auto beta_tensor = src_graph->CreateTensor(param_spec);
  auto gamma_tensor = src_graph->CreateTensor(param_spec);
  auto output_tensor = src_graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
    0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 4.0f, 1.0f, -1.0f, -1.0f, 2.0f, -1.0f, -2.0f, 1.0f, 4.0f
  };
  std::vector<float> beta = {0,0};
  std::vector<float> gamma = {1.0f,1.0f};
  std::vector<float> golden = {
    0.0f, -1.1470304f, 0.0f, -0.22940612f, 0.0f, -0.22940612f, 0.0f, 1.6058424f, 0.99995005f,
    -0.7337929f, -0.99995005f, 0.52413774f, -0.99995005f, -1.1531031f, 0.99995005f, 1.3627582f,
  };
  auto op = src_graph->CreateOperation<tim::vx::ops::InstanceNormalization>(1e-4f, tim::vx::DataLayout::CWHN);
  (*op).BindInputs({input_tensor, beta_tensor, gamma_tensor}).BindOutputs({output_tensor});
  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();

  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_beta = graph_io_map[src_graph->InputsTensor()[1]];
  auto infer_gamma = graph_io_map[src_graph->InputsTensor()[2]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_beta->CopyDataToTensor(beta.data(), beta.size() * sizeof(float));
  infer_gamma->CopyDataToTensor(gamma.data(), gamma.size() * sizeof(float));
  infer_graph->Run();

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Resize, bilinear_factor) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({1,2,2, 1});
    tim::vx::ShapeType output_shape({1,3,3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {1.0f, 1.0f, 2.0f, 2.0f};
    std::vector<float> golden = {1.0f, 1.0f, 1.0f, 1.6666666269302368f, 1.6666666269302368f,
                                    1.6666666269302368f, 2.0f, 2.0f, 2.0f};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Resize>(tim::vx::ResizeType::BILINEAR,
            1.7999999523162842f, false, false, 0, 0, tim::vx::DataLayout::CWHN);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    auto transform = tim::transform::LayoutInference(graph, ctx);
    auto infer_graph = transform.first;
    auto graph_io_map = transform.second;
    infer_graph->Compile();

    auto infer_input = graph_io_map[graph->InputsTensor()[0]];
    auto infer_output = graph_io_map[graph->OutputsTensor()[0]];

    infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
    infer_graph->Run();

    std::vector<float> output(golden.size());
    EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(Resize, bilinear_outputsize) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({1,2,2, 1});
    tim::vx::ShapeType output_shape({1,3,3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {1.0f, 1.0f, 2.0f, 2.0f};
    std::vector<float> golden = {1.0f, 1.0f, 1.0f, 1.6666666269302368f, 1.6666666269302368f,
                                    1.6666666269302368f, 2.0f, 2.0f, 2.0f};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Resize>(tim::vx::ResizeType::BILINEAR,
            0, false, false, 3, 3, tim::vx::DataLayout::CWHN);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    auto transform = tim::transform::LayoutInference(graph, ctx);
    auto infer_graph = transform.first;
    auto graph_io_map = transform.second;
    infer_graph->Compile();

    auto infer_input = graph_io_map[graph->InputsTensor()[0]];
    auto infer_output = graph_io_map[graph->OutputsTensor()[0]];

    infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
    infer_graph->Run();

    std::vector<float> output(golden.size());
    EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}

TEST(RoiAlign, nhwc) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 4, 4, 1});  //cwhn
  tim::vx::ShapeType regions_shape({4, 4});
  tim::vx::ShapeType batch_index_shape({4});
  tim::vx::ShapeType output_shape({1, 2, 2, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec regions_spec(tim::vx::DataType::FLOAT32, regions_shape,
                                   tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec batch_index_spec(tim::vx::DataType::INT32,
                                       batch_index_shape,
                                       tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  std::vector<float> input_data = {-10.0f, -1.0f, 4.0f,  -5.0f, -8.0f, -2.0f,
                                   9.0f,   1.0f,  7.0f,  -2.0f, 3.0f,  -7.0f,
                                   -2.0f,  10.0f, -3.0f, 5.0f};

  std::vector<float> regions_data = {2.0f, 2.0f, 4.0f, 4.0f, 0.0f, 0.0f,
                                     8.0f, 8.0f, 2.0f, 0.0f, 4.0f, 8.0f,
                                     0.0f, 2.0f, 8.0f, 4.0f};

  std::vector<int32_t> batch_index_data = {0, 0, 0, 0};

  std::vector<float> golden = {
      0.375f, 5.125f, -0.375f, 2.875f, -0.5f,    -0.3125f, 3.1875f, 1.125f,
      0.25f,  4.25f,  4.875f,  0.625f, -0.1875f, 1.125f,   0.9375f, -2.625f};

  auto input_tensor = src_graph->CreateTensor(input_spec);
  auto regions_tensor = src_graph->CreateTensor(regions_spec, regions_data.data());
  auto batch_index_tensor =
      src_graph->CreateTensor(batch_index_spec, batch_index_data.data());
  auto output_tensor = src_graph->CreateTensor(output_spec);

  auto roi_align = src_graph->CreateOperation<tim::vx::ops::RoiAlign>(
      2, 2, 2.0f, 2.0f, 4, 4, tim::vx::DataLayout::CWHN);
  (*roi_align)
      .BindInput(input_tensor)
      .BindInput(regions_tensor)
      .BindInput(batch_index_tensor)
      .BindOutput(output_tensor);

  // Do layout inference
  auto transform = tim::transform::LayoutInference(src_graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  infer_graph->Compile();

  auto infer_input = graph_io_map[src_graph->InputsTensor()[0]];
  auto infer_beta = graph_io_map[src_graph->InputsTensor()[1]];
  auto infer_gamma = graph_io_map[src_graph->InputsTensor()[2]];
  auto infer_output = graph_io_map[src_graph->OutputsTensor()[0]];

  infer_input->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(float));
  infer_beta->CopyDataToTensor(regions_data.data(), regions_data.size() * sizeof(float));
  infer_gamma->CopyDataToTensor(batch_index_data.data(), batch_index_data.size() * sizeof(float));
  infer_graph->Run();

  std::vector<float> output(golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}