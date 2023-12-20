#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"
#include "permute_vector.h"
#include "test_utils.h"

#include <algorithm>

#include "gtest/gtest.h"

TEST(RNNCell, layout_infer_align) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  float tolerance = ctx->hasSP() ? 1e-4f : 1e-5f;

  uint32_t input_size = 3, batch_size = 2, num_units = 4;

  tim::vx::ShapeType input_shape({input_size, batch_size});
  tim::vx::ShapeType weights_shape({input_size, num_units});
  tim::vx::ShapeType recurrent_weights_shape({num_units, num_units});
  tim::vx::ShapeType bias_shape({num_units});
  tim::vx::ShapeType state_in_shape({num_units, batch_size});
  tim::vx::ShapeType output_shape({num_units, batch_size});
  tim::vx::ShapeType state_out_shape({num_units, batch_size});
  tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weights_spec(tim::vx::DataType::FLOAT32, weights_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec recurrent_weights_spec(
      tim::vx::DataType::FLOAT32, recurrent_weights_shape,
      tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32, state_in_shape,
                                    tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec state_out_spec(tim::vx::DataType::UINT8, state_out_shape,
                                     tim::vx::TensorAttribute::OUTPUT, quant);

  std::vector<float> in_data = {
      0.12609188, 0.46347019, 0.89598465, 0.35867718, 0.36897406, 0.73463392,
  };
  std::vector<float> weights_data = {
      0.12609188, 0.46347019, 0.89598465, 0.35867718, 0.36897406, 0.73463392,
      0.12609188, 0.46347019, 0.89598465, 0.35867718, 0.36897406, 0.73463392,
  };
  std::vector<float> recurrent_weights_data = {
      -0.31930989, 0.37613347, 0.27901134, 0.36137494, -1.36916667, 0.38031587,
      0.21580373,  0.27072677, 1.01580888, 0.14943552, 1.15465137,  0.09784451,
      -1.02702999, 1.39296314, 0.15785322, 0.21931258,
  };
  std::vector<float> bias_data = {
      0.01580888,
      0.14943552,
      0.15465137,
      0.09784451,
  };
  std::vector<float> state_in_data = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> output_golden = {0.781534, 0.771447, 0.830002, 0.749713,
                                      0.711524, 0.74155,  0.77355,  0.717427};
  std::vector<uint8_t> state_out_golden = {
      217, 214, 231, 208, 198, 206, 215, 199,
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weights_tensor = graph->CreateTensor(weights_spec, weights_data.data());
  auto recurrent_weights_tensor = graph->CreateTensor(
      recurrent_weights_spec, recurrent_weights_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto state_in_tensor = graph->CreateTensor(state_in_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto state_out_tensor = graph->CreateTensor(state_out_spec);

  auto op = graph->CreateOperation<tim::vx::ops::RNNCell>(
      tim::vx::ops::RNNCell::ActivationType::kSIGMOID);
  (*op)
      .BindInputs({input_tensor, weights_tensor, bias_tensor, state_in_tensor,
                   recurrent_weights_tensor})
      .BindOutputs({output_tensor, state_out_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto infer_graph = transform.first;
  EXPECT_TRUE(infer_graph->Compile());
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_input_state = graph_io_map[graph->InputsTensor()[1]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  auto infer_output_state = graph_io_map[graph->OutputsTensor()[1]];

  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_input_state->CopyDataToTensor(state_in_data.data(),
                                      state_in_data.size() * sizeof(float));
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(output_golden.size());
  std::vector<uint8_t> state_out(state_out_golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(infer_output_state->CopyDataFromTensor(state_out.data()));
  EXPECT_TRUE(ArraysMatch(output_golden, output, tolerance));
  EXPECT_EQ(state_out_golden, state_out);
}

TEST(RNNCell, layout_infer_notalign) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();
  float tolerance = ctx->hasSP() ? 1e-4f : 1e-5f;

  uint32_t input_size = 3, batch_size = 2, num_units = 4;

  tim::vx::ShapeType input_shape({batch_size, input_size});  //input_pv={1,0}
  tim::vx::ShapeType weights_shape({input_size,num_units});
  tim::vx::ShapeType recurrent_weights_shape({num_units, num_units});
  tim::vx::ShapeType bias_shape({num_units});
  tim::vx::ShapeType state_in_shape({num_units, batch_size});
  tim::vx::ShapeType output_shape({num_units, batch_size});
  tim::vx::ShapeType state_out_shape({num_units, batch_size});
  tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec weights_spec(tim::vx::DataType::FLOAT32, weights_shape,
                                   tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec recurrent_weights_spec(
      tim::vx::DataType::FLOAT32, recurrent_weights_shape,
      tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec bias_spec(tim::vx::DataType::FLOAT32, bias_shape,
                                tim::vx::TensorAttribute::CONSTANT);
  tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32, state_in_shape,
                                    tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  tim::vx::TensorSpec state_out_spec(tim::vx::DataType::UINT8, state_out_shape,
                                     tim::vx::TensorAttribute::OUTPUT, quant);

  std::vector<float> in_data = {
      0.12609188, 0.35867718, 0.46347019, 0.36897406, 0.89598465, 0.73463392,
  };
  std::vector<float> weights_data = {
      0.12609188, 0.46347019, 0.89598465, 0.35867718, 0.36897406, 0.73463392,
      0.12609188, 0.46347019, 0.89598465, 0.35867718, 0.36897406, 0.73463392,
  };
  std::vector<float> recurrent_weights_data = {
      -0.31930989, 0.37613347, 0.27901134, 0.36137494, -1.36916667, 0.38031587,
      0.21580373,  0.27072677, 1.01580888, 0.14943552, 1.15465137,  0.09784451,
      -1.02702999, 1.39296314, 0.15785322, 0.21931258,
  };
  std::vector<float> bias_data = {
      0.01580888,
      0.14943552,
      0.15465137,
      0.09784451,
  };
  std::vector<float> state_in_data = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> output_golden = {0.781534, 0.771447, 0.830002, 0.749713,
                                      0.711524, 0.74155,  0.77355,  0.717427};
  std::vector<uint8_t> state_out_golden = {
      217, 214, 231, 208, 198, 206, 215, 199,
  };

  auto input_tensor = graph->CreateTensor(input_spec);
  auto weights_tensor = graph->CreateTensor(weights_spec, weights_data.data());
  auto recurrent_weights_tensor = graph->CreateTensor(
      recurrent_weights_spec, recurrent_weights_data.data());
  auto bias_tensor = graph->CreateTensor(bias_spec, bias_data.data());
  auto state_in_tensor = graph->CreateTensor(state_in_spec);
  auto output_tensor = graph->CreateTensor(output_spec);
  auto state_out_tensor = graph->CreateTensor(state_out_spec);
  std::map<std::shared_ptr<tim::vx::Tensor>,
           std::shared_ptr<tim::transform::IPermuteVector>>
      tensor_pv_map;
  std::shared_ptr<tim::transform::IPermuteVector> pv =
      std::make_shared<tim::transform::PermuteVector<2>>(
          std::initializer_list<uint32_t>({1U, 0U}));
  tensor_pv_map.insert({input_tensor, pv});

  auto op = graph->CreateOperation<tim::vx::ops::RNNCell>(
      tim::vx::ops::RNNCell::ActivationType::kSIGMOID);
  (*op)
      .BindInputs({input_tensor, weights_tensor, bias_tensor, state_in_tensor,
                   recurrent_weights_tensor})
      .BindOutputs({output_tensor, state_out_tensor});

  auto transform = tim::transform::LayoutInference(graph, ctx, tensor_pv_map);
  auto infer_graph = transform.first;
  EXPECT_TRUE(infer_graph->Compile());
  auto graph_io_map = transform.second;
  auto infer_input = graph_io_map[graph->InputsTensor()[0]];
  auto infer_input_state = graph_io_map[graph->InputsTensor()[1]];
  auto infer_output = graph_io_map[graph->OutputsTensor()[0]];
  auto infer_output_state = graph_io_map[graph->OutputsTensor()[1]];

  infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float));
  infer_input_state->CopyDataToTensor(state_in_data.data(),
                                      state_in_data.size() * sizeof(float));
  EXPECT_TRUE(infer_graph->Run());

  std::vector<float> output(output_golden.size());
  std::vector<uint8_t> state_out(state_out_golden.size());
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  EXPECT_TRUE(infer_output_state->CopyDataFromTensor(state_out.data()));
  EXPECT_TRUE(ArraysMatch(output_golden, output, tolerance));
  EXPECT_EQ(state_out_golden, state_out);
}