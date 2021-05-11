#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/transform/layout_inference.h"

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
      std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);
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