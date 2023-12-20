#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"

TEST(Pad, layout_inference) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({32, 112, 112, 1});  //cwhn
  tim::vx::ShapeType kernel_shape({32, 2, 2, 32});    //iwho
  //   tim::vx::ShapeType conv2dout_shape({32, 111, 111, 1});  //iwho
  tim::vx::ShapeType output_shape({32, 112, 112, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec kernel_spec(tim::vx::DataType::FLOAT32, kernel_shape,
                                  tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec conv2dout_spec(tim::vx::DataType::FLOAT32, {0, 0, 0, 0},
                                     tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto conv2dout_tensor = graph->CreateTensor(conv2dout_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data;
  for (uint32_t i = 0; i < 32 * 112 * 112; ++i) {
    in_data.push_back(0.5);
  };
  std::vector<float> kernel_data;
  for (uint32_t i = 0; i < 4 * 32 * 32; ++i) {
    kernel_data.push_back(0.5);
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

  std::vector<uint32_t> front_size = {0, 0, 0, 0};
  std::vector<uint32_t> back_size = {0, 1, 1, 0};

  auto op2 =
      graph->CreateOperation<tim::vx::ops::Pad>(front_size, back_size, 0);
  (*op2).BindInputs({conv2dout_tensor}).BindOutputs({output_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));

  EXPECT_TRUE(infer_graph->Compile());
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(32*112*112);
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
}
