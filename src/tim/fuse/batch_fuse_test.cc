#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/pad.h"
// #include "tim/vx/ops/pad_v2.h"
#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/transpose.h"
#include "tim/fuse/batch_fuse.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"

// class BatchFuse_resnet_pad
//     : public testing::TestWithParam<std::tuple<uint32_t>> {
//  public:
//   void Run_resnet(uint32_t batch) {
//     auto ctx = tim::vx::Context::Create();
//     auto src_graph = ctx->CreateGraph();

//     tim::vx::ShapeType input_shape({3, 224, 224, batch});  //cwhn
//     tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
//                                    tim::vx::TensorAttribute::INPUT);

//     // tim::vx::ShapeType pad_shape({230, 230, 3, batch});  //whcn
//     // tim::vx::TensorSpec pad_spec(tim::vx::DataType::FLOAT32, pad_shape,
//     //                                tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType input_0_shape({3, 230, 230, batch});  //cwhn
//     tim::vx::TensorSpec input_0_spec(tim::vx::DataType::FLOAT32, input_0_shape,
//                                      tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType kernel_0_shape({3, 7, 7, 64});  //iwno
//     tim::vx::TensorSpec kernel_0_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_0_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType output_0_shape({64, 112, 112, batch});  //cwhn
//     tim::vx::TensorSpec output_0_spec(tim::vx::DataType::FLOAT32,
//                                       output_0_shape,
//                                       tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType input_1_shape({64, 56, 56, batch});  //cwhn
//     tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, input_1_shape,
//                                      tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType kernel_1_shape({64, 2, 2, 64});  //iwho
//     tim::vx::TensorSpec kernel_1_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_1_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType output_1_shape({64, 56, 56, batch});
//     tim::vx::TensorSpec output_1_spec(tim::vx::DataType::FLOAT32,
//                                       output_1_shape,
//                                       tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType relu_1_shape({64, 56, 56, batch});
//     tim::vx::TensorSpec relu_1_spec(tim::vx::DataType::FLOAT32, relu_1_shape,
//                                     tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType kernel_2_shape({64, 3, 3, 64});  //iwho
//     tim::vx::TensorSpec kernel_2_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_2_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType output_2_shape({64, 56, 56, batch});
//     tim::vx::TensorSpec output_2_spec(tim::vx::DataType::FLOAT32,
//                                       output_2_shape,
//                                       tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType relu_2_shape({64, 56, 56, batch});
//     tim::vx::TensorSpec relu_2_spec(tim::vx::DataType::FLOAT32, relu_2_shape,
//                                     tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType kernel_3_shape({64, 2, 2, 256});  //iwho
//     tim::vx::TensorSpec kernel_3_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_3_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType output_3_shape({256, 56, 56, batch});
//     tim::vx::TensorSpec output_3_spec(tim::vx::DataType::FLOAT32,
//                                       output_3_shape,
//                                       tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType kernel_4_shape({64, 2, 2, 256});  //iwho
//     tim::vx::TensorSpec kernel_4_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_4_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType output_4_shape({256, 56, 56, batch});
//     tim::vx::TensorSpec output_4_spec(tim::vx::DataType::FLOAT32,
//                                       output_4_shape,
//                                       tim::vx::TensorAttribute::TRANSIENT);

//     tim::vx::ShapeType bias_1_shape({1});
//     tim::vx::TensorSpec bias_1_spec(tim::vx::DataType::FLOAT32, bias_1_shape,
//                                     tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType add_output_shape({256, 56, 56, batch});
//     tim::vx::TensorSpec add_output_spec(tim::vx::DataType::FLOAT32,
//                                         add_output_shape,
//                                         tim::vx::TensorAttribute::TRANSIENT);
//     tim::vx::ShapeType add_output_1_shape({256, 56, 56, batch});
//     tim::vx::TensorSpec add_output_1_spec(tim::vx::DataType::FLOAT32,
//                                           add_output_1_shape,
//                                           tim::vx::TensorAttribute::OUTPUT);

//     std::vector<float> kernel_0_data(64 * 7 * 7 * 3, 1.0f);
//     std::vector<float> kernel_1_data(64 * 64 * 4, 1.0f);
//     std::vector<float> kernel_2_data(64 * 9 * 64, 1.0f);
//     std::vector<float> kernel_3_data(64 * 256 * 4, 1.0f);
//     std::vector<float> kernel_4_data(64 * 256 * 4, 1.0f);
//     std::vector<float> bias_data_0 = {64.f};
//     std::vector<float> bias_data_1 = {64.f};
//     std::vector<float> bias_data_2 = {64.f};
//     std::vector<float> bias_data_3 = {256.f};
//     std::vector<float> bias_data_4 = {256.f};

//     auto input = src_graph->CreateTensor(input_spec);

//     auto kernel_0 =
//         src_graph->CreateTensor(kernel_0_spec, kernel_0_data.data());
//     auto kernel_1 =
//         src_graph->CreateTensor(kernel_1_spec, kernel_1_data.data());
//     auto kernel_2 =
//         src_graph->CreateTensor(kernel_2_spec, kernel_2_data.data());
//     auto kernel_3 =
//         src_graph->CreateTensor(kernel_3_spec, kernel_3_data.data());
//     auto kernel_4 =
//         src_graph->CreateTensor(kernel_4_spec, kernel_4_data.data());
//     auto input_0 = src_graph->CreateTensor(input_0_spec);
//     auto input_1 = src_graph->CreateTensor(input_1_spec);
//     auto output_0 = src_graph->CreateTensor(output_0_spec);
//     auto output_1 = src_graph->CreateTensor(output_1_spec);
//     auto output_2 = src_graph->CreateTensor(output_2_spec);
//     auto output_3 = src_graph->CreateTensor(output_3_spec);
//     auto output_4 = src_graph->CreateTensor(output_4_spec);
//     auto relu_1 = src_graph->CreateTensor(relu_1_spec);
//     auto relu_2 = src_graph->CreateTensor(relu_2_spec);
//     auto bias_0 = src_graph->CreateTensor(bias_1_spec, bias_data_0.data());
//     auto bias_1 = src_graph->CreateTensor(bias_1_spec, bias_data_1.data());
//     auto bias_2 = src_graph->CreateTensor(bias_1_spec, bias_data_2.data());
//     auto bias_3 = src_graph->CreateTensor(bias_1_spec, bias_data_3.data());
//     auto bias_4 = src_graph->CreateTensor(bias_1_spec, bias_data_4.data());
//     auto add_output = src_graph->CreateTensor(add_output_spec);
//     auto add_relu = src_graph->CreateTensor(add_output_1_spec);

//     std::vector<uint32_t> front = {0, 3, 3, 0};
//     std::vector<uint32_t> back = {0, 3, 3, 0};
//     auto pad_op = src_graph->CreateOperation<tim::vx::ops::Pad>(
//         front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
//     auto conv2d_0 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_0_shape[3], tim::vx::PadType::VALID,
//         std::array<uint32_t, 2>({kernel_1_shape[0], kernel_1_shape[1]}),
//         std::array<uint32_t, 2>({2, 2}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);
//     auto conv2d_1 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_1_shape[3], tim::vx::PadType::SAME,
//         std::array<uint32_t, 2>({kernel_1_shape[0], kernel_1_shape[1]}),
//         std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);
//     auto conv2d_2 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_2_shape[3], tim::vx::PadType::SAME,
//         std::array<uint32_t, 2>({kernel_2_shape[0], kernel_2_shape[1]}),
//         std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);

//     auto conv2d_3 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_3_shape[3], tim::vx::PadType::SAME,
//         std::array<uint32_t, 2>({kernel_3_shape[0], kernel_3_shape[1]}),
//         std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);

//     auto conv2d_4 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_4_shape[3], tim::vx::PadType::SAME,
//         std::array<uint32_t, 2>({kernel_4_shape[0], kernel_4_shape[1]}),
//         std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);

//     auto relu_1_op = src_graph->CreateOperation<tim::vx::ops::Relu>();
//     auto relu_2_op = src_graph->CreateOperation<tim::vx::ops::Relu>();
//     auto relu_add_op = src_graph->CreateOperation<tim::vx::ops::Relu>();

//     auto pool2d = src_graph->CreateOperation<tim::vx::ops::Pool2d>(
//         tim::vx::PoolType::MAX, tim::vx::PadType::SAME,
//         std::array<uint32_t, 2>({3, 3}), std::array<uint32_t, 2>({2, 2}),
//         tim::vx::RoundType::FLOOR, tim::vx::DataLayout::CWHN);

//     std::vector<int32_t> axis = {0, 1};
//     auto add_op = src_graph->CreateOperation<tim::vx::ops::Add>();

//     (*pad_op).BindInput(input).BindOutput(input_0);
//     (*conv2d_0).BindInputs({input_0, kernel_0, bias_0}).BindOutput(output_0);
//     (*pool2d).BindInput(output_0).BindOutput(input_1);
//     (*conv2d_4).BindInputs({input_1, kernel_4, bias_4}).BindOutput(output_4);

//     (*conv2d_1).BindInputs({input_1, kernel_1, bias_1}).BindOutput(output_1);
//     (*relu_1_op).BindInput(output_1).BindOutput(relu_1);
//     (*conv2d_2).BindInputs({relu_1, kernel_2, bias_2}).BindOutput(output_2);
//     (*relu_2_op).BindInput(output_2).BindOutput(relu_2);
//     (*conv2d_3).BindInputs({relu_2, kernel_3, bias_3}).BindOutput(output_3);

//     (*add_op).BindInputs({output_3, output_4}).BindOutput(add_output);
//     (*relu_add_op).BindInput(add_output).BindOutput(add_relu);
//     // (*reduce_op).BindInput(add_relu).BindOutput(reduce_output);

//     auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
//     // const std::shared_ptr<tim::vx::Graph> layout_const_ = layout_infer.first;
//     auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 1);
//     auto batch_fuse_graph = batchfuse.first;
//     auto graph_io_map = batchfuse.second;

//     // auto batch_fuse_graph = layout_infer.first;
//     // auto graph_io_map = layout_infer.second;
//     batch_fuse_graph->Compile();
//     std::vector<float> input_data(224 * 224 * 3 * batch, 2.f);

//     auto batch_fuse_input_0 =
//         graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
//     auto batch_fuse_output_0 =
//         graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
//     auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
//     auto batch_fuse_output =
//         graph_io_map[layout_infer.first->OutputsTensor()[0]];

//     auto batch_fuse_input_shape = batch_fuse_input->GetShape();
//     batch_fuse_input->CopyDataToTensor(input_data.data(),
//                                        input_data.size() * sizeof(float));
//     batch_fuse_graph->Run();
//     std::vector<float> out_data;
//     auto batch_fuse_out_shape = batch_fuse_output->GetShape();
//     out_data.resize(batch_fuse_out_shape[0] * batch_fuse_out_shape[1] *
//                     batch_fuse_out_shape[2] * batch_fuse_out_shape[3]);
//     batch_fuse_output->CopyDataFromTensor(out_data.data());
//     for (int i = 0; i < out_data.size(); ++i) {
//       std::cout << out_data[i] << " ";
//     }
//     std::cout << std::endl;
//     tim::vx::ShapeType expect_shape({256, 56, 56, batch});
//     EXPECT_EQ(batch_fuse_out_shape, expect_shape);
//   }
// };

// TEST_P(BatchFuse_resnet_pad, resnet) {
//   uint32_t batch;
//   std::tie(batch) = GetParam();
//   Run_resnet(batch);
// }

// INSTANTIATE_TEST_SUITE_P(Float_test, BatchFuse_resnet_pad,
//                          testing::Combine(testing::Values(1, 2, 4, 8, 9,
//                                                           16)  //batch
//                                           ));

// class BatchFuse_conv2d : public testing::TestWithParam<std::tuple<uint32_t>> {
//  public:
//   void Run_resnet(uint32_t batch) {
//     auto ctx = tim::vx::Context::Create();
//     auto src_graph = ctx->CreateGraph();

//     tim::vx::ShapeType input_shape({1, 4, 4, batch});  //cwhn
//     tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
//                                    tim::vx::TensorAttribute::INPUT);

//     tim::vx::ShapeType kernel_0_shape({1, 2, 2, 1});  //iwho
//     tim::vx::TensorSpec kernel_0_spec(tim::vx::DataType::FLOAT32,
//                                       kernel_0_shape,
//                                       tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType bias_0_shape({1});
//     tim::vx::TensorSpec bias_0_spec(tim::vx::DataType::FLOAT32, bias_0_shape,
//                                     tim::vx::TensorAttribute::CONSTANT);

//     tim::vx::ShapeType input_0_shape({1, 3, 3, batch});  //cwhn
//     tim::vx::TensorSpec input_0_spec(tim::vx::DataType::FLOAT32, input_0_shape,
//                                      tim::vx::TensorAttribute::OUTPUT);

//     auto input = src_graph->CreateTensor(input_spec);
//     auto input_0 = src_graph->CreateTensor(input_0_spec);

//     std::vector<float> kernel_0_data = {1.f, 1.f, 1.f, 1.f};
//     std::vector<float> bias_0_data = {0.f};
//     auto kernel_0 =
//         src_graph->CreateTensor(kernel_0_spec, kernel_0_data.data());
//     auto bias_0 = src_graph->CreateTensor(bias_0_spec, bias_0_data.data());

//     auto conv2d_0 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
//         kernel_0_shape[3], tim::vx::PadType::VALID,
//         std::array<uint32_t, 2>({kernel_0_shape[0], kernel_0_shape[1]}),
//         std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
//         std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);

//     (*conv2d_0).BindInputs({input, kernel_0, bias_0}).BindOutput(input_0);
//     auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
//     auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, batch);
//     auto batch_fuse_graph = batchfuse.first;
//     auto graph_io_map = batchfuse.second;

//     batch_fuse_graph->Compile();
//     // std::vector<float> input_data(16 * batch, 2.f);
//     std::vector<float> input_data(16 * batch);
//     for (int i = 0; i < input_data.size(); ++i) {
//       input_data[i] = i;
//     }
//     auto batch_fuse_input_0 =
//         graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
//     auto batch_fuse_output_0 =
//         graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
//     auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
//     auto batch_fuse_output =
//         graph_io_map[layout_infer.first->OutputsTensor()[0]];

//     auto batch_fuse_input_shape = batch_fuse_input->GetShape();
//     batch_fuse_input->CopyDataToTensor(input_data.data(),
//                                        input_data.size() * sizeof(float));
//     batch_fuse_graph->Run();
//     std::vector<float> out_data;
//     auto batch_fuse_out_shape = batch_fuse_output->GetShape();
//     out_data.resize(batch_fuse_out_shape[0] * batch_fuse_out_shape[1] *
//                     batch_fuse_out_shape[2] * batch_fuse_out_shape[3]);
//     batch_fuse_output->CopyDataFromTensor(out_data.data());
//     for (int i = 0; i < out_data.size(); ++i) {
//       std::cout << out_data[i] << " ";
//     }
//     std::cout << std::endl;
//   }
// };

// TEST_P(BatchFuse_conv2d, resnet) {
//   uint32_t batch;
//   std::tie(batch) = GetParam();
//   Run_resnet(batch);
// }

// INSTANTIATE_TEST_SUITE_P(Float_test, BatchFuse_conv2d,
//                          testing::Combine(testing::Values(1, 2, 4, 8, 9,
//                                                           16)  //batch
//                                           ));


TEST(BatchFuse, uint8_pad_1324) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 4});
  tim::vx::ShapeType output_shape({1, 7, 4, 4});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<uint8_t> input_data = {
      1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
  };

  std::vector<uint32_t> front = {0, 1, 0, 0};
  std::vector<uint32_t> back = {0, 3, 2, 0};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);
  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(transform.first, ctx, 4);
  auto infer_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;
  auto infer_input = graph_io_map[transform.second[graph->InputsTensor()[0]]];
  auto infer_output = graph_io_map[transform.second[graph->OutputsTensor()[0]]];

  EXPECT_TRUE(infer_graph->Compile());
  infer_input->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(uint8_t));
  EXPECT_TRUE(infer_graph->Run());
  std::vector<uint8_t> output(7 * 16);
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  std::vector<uint8_t> golden = {0, 1, 1, 1, 0, 0, 0, 
                                0, 1, 1, 1, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0,
                                0, 2, 2, 2, 0, 0, 0, 
                                0, 2, 2, 2, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 3, 3, 3, 0, 0, 0, 
                                0, 3, 3, 3, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 4, 4, 4, 0, 0, 0, 
                                0, 4, 4, 4, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(output, golden);
  
}

TEST(BatchFuse, int8_pad_1321) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 1});
  tim::vx::ShapeType output_shape({1, 7, 4, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int8_t> input_data = {1, 1, 1, 
                                    1, 1, 1};
  std::vector<uint32_t> front = {0, 1, 0, 0};
  std::vector<uint32_t> back = {0, 3, 2, 0};
  auto op = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*op).BindInput(input_tensor).BindOutput(output_tensor);
  auto transform = tim::transform::LayoutInference(graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(transform.first, ctx, 1);
  auto infer_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;
  auto infer_input = graph_io_map[transform.second[graph->InputsTensor()[0]]];
  auto infer_output = graph_io_map[transform.second[graph->OutputsTensor()[0]]];

  EXPECT_TRUE(infer_graph->Compile());
  infer_input->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(int8_t));
  EXPECT_TRUE(infer_graph->Run());
  std::vector<int8_t> output(28);
  EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));
  std::vector<int8_t> golden = {0, 1, 1, 1, 0, 0, 0, 
                                0, 1, 1, 1, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(output, golden);
}

TEST(BatchFuse, int8_transpose_pad_1324) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 4});
  tim::vx::ShapeType trans_shape({3, 2, 1, 4});
  tim::vx::ShapeType output_shape({7, 4, 1, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec trans_spec(tim::vx::DataType::INT8, trans_shape,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto trans_tensor = graph->CreateTensor(trans_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<int8_t> input_data = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4};

  std::vector<uint32_t> front = {1, 0, 0, 0};
  std::vector<uint32_t> back = {3, 2, 0, 0};
  std::vector<uint32_t> perm_1 = {1, 2, 0, 3};
  auto transpose = graph->CreateOperation<tim::vx::ops::Transpose>(perm_1);
  auto pad = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*transpose).BindInput(input_tensor).BindOutput(trans_tensor);
  (*pad).BindInput(trans_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(int8_t));
  EXPECT_TRUE(graph->Run());
  std::vector<int8_t> output(7 * 16);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  std::vector<int8_t> golden = {0, 1, 1, 1, 0, 0, 0, 
                                0, 1, 1, 1, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0,
                                0, 2, 2, 2, 0, 0, 0, 
                                0, 2, 2, 2, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 3, 3, 3, 0, 0, 0, 
                                0, 3, 3, 3, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 4, 4, 4, 0, 0, 0, 
                                0, 4, 4, 4, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(output, golden);
}

TEST(BatchFuse, float_transpose_pad_1324) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 4});
  tim::vx::ShapeType trans_shape({3, 2, 1, 4});
  tim::vx::ShapeType output_shape({7, 4, 1, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec trans_spec(tim::vx::DataType::FLOAT32, trans_shape,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto trans_tensor = graph->CreateTensor(trans_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {1.f, 1.f, 1.f, -1.f, -1.f, -1.f, 2.f, 2.f,
                                   2.f, -2.f, -2.f, -2.f, 3.f, 3.f, 3.f, -3.f,
                                   -3.f, -3.f, 4.f, 4.f, 4.f, -4.f, -4.f, -4.f};
  std::vector<uint32_t> front = {1, 0, 0, 0};
  std::vector<uint32_t> back = {3, 2, 0, 0};
  std::vector<uint32_t> perm_1 = {1, 2, 0, 3};
  auto transpose = graph->CreateOperation<tim::vx::ops::Transpose>(perm_1);
  auto pad = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*transpose).BindInput(input_tensor).BindOutput(trans_tensor);
  (*pad).BindInput(trans_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  input_tensor->CopyDataToTensor(input_data.data(), input_data.size() * sizeof(float));

  EXPECT_TRUE(graph->Run());
  std::vector<float> output(7 * 16);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  std::vector<float> golden = {0, 1, 1, 1, 0, 0, 0, 
                                0, -1, -1, -1, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0,
                                0, 2, 2, 2, 0, 0, 0, 
                                0, -2, -2, -2, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 3, 3, 3, 0, 0, 0, 
                                0, -3, -3, -3, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 4, 4, 4, 0, 0, 0, 
                                0, -4, -4, -4, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0, 
                                0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(output, golden);
}

TEST(BatchFuse, float_transpose_pad_1321) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 1});
  tim::vx::ShapeType trans_shape({3, 2, 1, 1});
  tim::vx::ShapeType output_shape({7, 4, 1, 1});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec trans_spec(tim::vx::DataType::FLOAT32, trans_shape,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto trans_tensor = graph->CreateTensor(trans_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> input_data = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  std::vector<uint32_t> front = {1, 0, 0, 0};
  std::vector<uint32_t> back = {3, 2, 0, 0};
  std::vector<uint32_t> perm_1 = {1, 2, 0, 3};
  auto transpose = graph->CreateOperation<tim::vx::ops::Transpose>(perm_1);
  auto pad = graph->CreateOperation<tim::vx::ops::Pad>(
      front, back, tim::vx::ops::Pad::PAD_MODE_CONSTANT);
  (*transpose).BindInput(input_tensor).BindOutput(trans_tensor);
  (*pad).BindInput(trans_tensor).BindOutput(output_tensor);

  EXPECT_TRUE(graph->Compile());
  input_tensor->CopyDataToTensor(input_data.data());

  EXPECT_TRUE(graph->Run());
  std::vector<float> output(28);
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  std::vector<float> gloden = {0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(output, gloden);
}

TEST(BatchFuse, reduce_float_1324) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 3, 2, 4});
  tim::vx::ShapeType out_shape({1, 1, 1, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32, out_shape,
                               tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(out_spec);

  std::vector<float> input_data = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f,
                                   2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                   3.f, 3.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};

  std::vector<int32_t> axis = {1, 2};
  auto reduce_op = graph->CreateOperation<tim::vx::ops::ReduceMean>(axis, true);
  (*reduce_op).BindInput(input_tensor).BindOutput(output_tensor);
  auto layout_infer = tim::transform::LayoutInference(graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  EXPECT_TRUE(batch_fuse_graph->Compile());
  auto batch_fuse_input_0 =
      graph_io_map[layout_infer.second[graph->InputsTensor()[0]]];
  auto batch_fuse_output_0 =
      graph_io_map[layout_infer.second[graph->OutputsTensor()[0]]];
  auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input->GetShape();
  batch_fuse_input->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));

  EXPECT_TRUE(batch_fuse_graph->Run());
  std::vector<float> output(4);
  EXPECT_TRUE(batch_fuse_output->CopyDataFromTensor(output.data()));
  std::vector<float> golden = {1, 2, 3, 4};
  EXPECT_EQ(output, golden);
}

TEST(BatchFuse, conv2d_float_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 4, 4, 4});  //cwhn
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);

  tim::vx::ShapeType kernel_0_shape({1, 2, 2, 1});  //iwho
  tim::vx::TensorSpec kernel_0_spec(tim::vx::DataType::FLOAT32, kernel_0_shape,
                                    tim::vx::TensorAttribute::CONSTANT);

  tim::vx::ShapeType bias_0_shape({1});
  tim::vx::TensorSpec bias_0_spec(tim::vx::DataType::FLOAT32, bias_0_shape,
                                  tim::vx::TensorAttribute::CONSTANT);

  tim::vx::ShapeType input_0_shape({1, 3, 3, 4});  //cwhn
  tim::vx::TensorSpec input_0_spec(tim::vx::DataType::FLOAT32, input_0_shape,
                                   tim::vx::TensorAttribute::OUTPUT);

  auto input = src_graph->CreateTensor(input_spec);
  auto input_0 = src_graph->CreateTensor(input_0_spec);

  std::vector<float> kernel_0_data = {1, 1, 1, 1};
  std::vector<float> bias_0_data = {0};
  auto kernel_0 = src_graph->CreateTensor(kernel_0_spec, kernel_0_data.data());
  auto bias_0 = src_graph->CreateTensor(bias_0_spec, bias_0_data.data());

  auto conv2d_0 = src_graph->CreateOperation<tim::vx::ops::Conv2d>(
      kernel_0_shape[3], tim::vx::PadType::VALID,
      std::array<uint32_t, 2>({kernel_0_shape[0], kernel_0_shape[1]}),
      std::array<uint32_t, 2>({1, 1}), std::array<uint32_t, 2>({0, 0}),
      std::array<uint32_t, 4>({0, 0, 0, 0}), 0, tim::vx::DataLayout::CWHN);

  (*conv2d_0).BindInputs({input, kernel_0, bias_0}).BindOutput(input_0);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  std::vector<float> input_data(16 * 4);
  for (uint i = 0; i < input_data.size(); ++i) {
    input_data[i] = i;
  }
  auto batch_fuse_input_0 =
      graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  auto batch_fuse_output_0 =
      graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input->GetShape();
  batch_fuse_input->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));
  batch_fuse_graph->Run();
  std::vector<float> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(batch_fuse_out_shape[0] * batch_fuse_out_shape[1] *
                  batch_fuse_out_shape[2] * batch_fuse_out_shape[3]);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
}

TEST(BatchFuse, pool_uint8_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 4, 4, 4});  //cwhn
  tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                 tim::vx::TensorAttribute::INPUT);

  tim::vx::ShapeType output_0_shape({1, 3, 3, 4});
  tim::vx::TensorSpec output_0_spec(tim::vx::DataType::UINT8, output_0_shape,
                                    tim::vx::TensorAttribute::OUTPUT);

  auto input = src_graph->CreateTensor(input_spec);
  auto output_0 = src_graph->CreateTensor(output_0_spec);

  auto pool2d = src_graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, tim::vx::PadType::VALID,
      std::array<uint32_t, 2>({2, 2}), std::array<uint32_t, 2>({1, 1}),
      tim::vx::RoundType::FLOOR, tim::vx::DataLayout::CWHN);

  (*pool2d).BindInput(input).BindOutput(output_0);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  std::vector<uint8_t> input_data(16 * 4);
  for (uint i = 0; i < input_data.size(); ++i) {
    input_data[i] = i;
  }

  auto batch_fuse_input_0 =
      graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  auto batch_fuse_output_0 =
      graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input->GetShape();
  batch_fuse_input->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(uint8_t));

  batch_fuse_graph->Run();
  std::vector<uint8_t> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(batch_fuse_out_shape[0] * batch_fuse_out_shape[1] *
                  batch_fuse_out_shape[2] * batch_fuse_out_shape[3]);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
  std::vector<uint8_t> golden = {5, 6, 7, 9, 10, 11, 13, 14, 15, 21, 22, 23, 25, 26, 27, 29, 30, 31, 37, 38, 39, 41, 42, 43, 45, 46, 47, 53, 54, 55, 57, 58, 59, 61, 62, 63};
  EXPECT_EQ(out_data, golden);
}

TEST(BatchFuse, reshape_float_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({1, 4, 4, 4});  //cwhn
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);

  tim::vx::ShapeType output_0_shape({16, 4});
  tim::vx::TensorSpec output_0_spec(tim::vx::DataType::FLOAT32, output_0_shape,
                                    tim::vx::TensorAttribute::OUTPUT);

  auto input = src_graph->CreateTensor(input_spec);
  auto output_0 = src_graph->CreateTensor(output_0_spec);

  auto reshape_op =
      src_graph->CreateOperation<tim::vx::ops::Reshape>(output_0_shape);

  (*reshape_op).BindInput(input).BindOutput(output_0);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  std::vector<float> input_data(16 * 4);
  for (uint i = 0; i < input_data.size(); ++i) {
    input_data[i] = i;
  }

  auto batch_fuse_input_0 =
      graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  auto batch_fuse_output_0 =
      graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input->GetShape();
  batch_fuse_input->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));

  batch_fuse_graph->Run();
  std::vector<float> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(16 * 4);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
}

TEST(BatchFuse, transpose_float_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_shape({2, 3, 2, 4});
  tim::vx::ShapeType trans_shape({3, 2, 2, 4});

  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, input_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec trans_spec(tim::vx::DataType::FLOAT32, trans_shape,
                                 tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = src_graph->CreateTensor(input_spec);
  auto trans_tensor = src_graph->CreateTensor(trans_spec);

  std::vector<float> input_data = {
      1.f, 1.f, 1.f, 1.f, 1.f, 1.f, -1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
      2.f, 2.f, 2.f, 2.f, 2.f, 2.f, -2.f, -2.f, -2.f, -2.f, -2.f, -2.f,
      3.f, 3.f, 3.f, 3.f, 3.f, 3.f, -3.f, -3.f, -3.f, -3.f, -3.f, -3.f,
      4.f, 4.f, 4.f, 4.f, 4.f, 4.f, -4.f, -4.f, -4.f, -4.f, -4.f, -4.f};

  std::vector<uint32_t> perm_1 = {1, 2, 0, 3};
  auto transpose = src_graph->CreateOperation<tim::vx::ops::Transpose>(perm_1);
  (*transpose).BindInput(input_tensor).BindOutput(trans_tensor);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  auto batch_fuse_input_0 =
      graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  auto batch_fuse_output_0 =
      graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input->GetShape();
  batch_fuse_input->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));

  batch_fuse_graph->Run();
  std::vector<float> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(48);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
  std::vector<float> golden = {1, 1, 1, -1, -1, -1, 
                              1, 1, 1, -1, -1, -1, 
                              2, 2, 2, -2, -2, -2, 
                              2, 2, 2, -2, -2, -2, 
                              3, 3, 3, -3, -3, -3, 
                              3, 3, 3, -3, -3, -3, 
                              4, 4, 4, -4, -4, -4, 
                              4, 4, 4, -4, -4, -4};
  EXPECT_EQ(out_data, golden);
}

TEST(BatchFuse, add_float_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_1_shape({1, 3, 2, 4});
  tim::vx::ShapeType input_2_shape({1, 3, 2, 4});
  tim::vx::ShapeType output_shape({1, 3, 2, 4});

  tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, input_1_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input_2_spec(tim::vx::DataType::FLOAT32, input_1_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                 tim::vx::TensorAttribute::OUTPUT);

  auto input_1_tensor = src_graph->CreateTensor(input_1_spec);
  auto input_2_tensor = src_graph->CreateTensor(input_2_spec);
  auto output_tensor = src_graph->CreateTensor(output_spec);

  std::vector<float> input_data = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f,
                                   2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                   3.f, 3.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};

  
  auto add= src_graph->CreateOperation<tim::vx::ops::Add>();
  (*add).BindInputs({input_1_tensor, input_2_tensor}).BindOutput(output_tensor);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  // auto batch_fuse_input_0 =
  //     graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  // auto batch_fuse_output_0 =
  //     graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input_1 = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_input_2 = graph_io_map[layout_infer.first->InputsTensor()[1]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input_1->GetShape();
  batch_fuse_input_1->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));
  batch_fuse_input_2->CopyDataToTensor(input_data.data(),
                                     input_data.size() * sizeof(float));

  batch_fuse_graph->Run();
  std::vector<float> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(24);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
  std::vector<float> golden = {2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8};
  EXPECT_EQ(out_data, golden);
}

TEST(BatchFuse, relu_float_1444) {
  auto ctx = tim::vx::Context::Create();
  auto src_graph = ctx->CreateGraph();

  tim::vx::ShapeType input_1_shape({1, 3, 2, 4});
  tim::vx::ShapeType input_2_shape({1, 3, 2, 4});
  tim::vx::ShapeType output_shape({1, 3, 2, 4});

  tim::vx::TensorSpec input_1_spec(tim::vx::DataType::FLOAT32, input_1_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec input_2_spec(tim::vx::DataType::FLOAT32, input_1_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec add_spec(tim::vx::DataType::FLOAT32, output_shape,
                                 tim::vx::TensorAttribute::TRANSIENT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, output_shape,
                                 tim::vx::TensorAttribute::OUTPUT);

  auto input_1_tensor = src_graph->CreateTensor(input_1_spec);
  auto input_2_tensor = src_graph->CreateTensor(input_2_spec);
  auto add_tensor = src_graph->CreateTensor(add_spec);
  auto output_tensor = src_graph->CreateTensor(output_spec);

  std::vector<float> input_data_1 = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 2.f,
                                   2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                   3.f, 3.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};

  std::vector<float> input_data_2 = {-2.f, -2.f, -2.f, -2.f, -2.f, -2.f, -2.f, -2.f,
                                   2.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 3.f,
                                   3.f, 3.f, 4.f, 4.f, 4.f, 4.f, 4.f, 4.f};

  auto add= src_graph->CreateOperation<tim::vx::ops::Add>();
  auto relu = src_graph->CreateOperation<tim::vx::ops::Relu>();
  (*add).BindInputs({input_1_tensor, input_2_tensor}).BindOutput(add_tensor);
  (*relu).BindInput(add_tensor).BindOutput(output_tensor);
  auto layout_infer = tim::transform::LayoutInference(src_graph, ctx);
  auto batchfuse = tim::fuse::BatchFuse(layout_infer.first, ctx, 4);
  auto batch_fuse_graph = batchfuse.first;
  auto graph_io_map = batchfuse.second;

  batch_fuse_graph->Compile();
  // auto batch_fuse_input_0 =
  //     graph_io_map[layout_infer.second[src_graph->InputsTensor()[0]]];
  // auto batch_fuse_output_0 =
  //     graph_io_map[layout_infer.second[src_graph->OutputsTensor()[0]]];
  auto batch_fuse_input_1 = graph_io_map[layout_infer.first->InputsTensor()[0]];
  auto batch_fuse_input_2 = graph_io_map[layout_infer.first->InputsTensor()[1]];
  auto batch_fuse_output = graph_io_map[layout_infer.first->OutputsTensor()[0]];

  auto batch_fuse_input_shape = batch_fuse_input_1->GetShape();
  batch_fuse_input_1->CopyDataToTensor(input_data_1.data(),
                                     input_data_1.size() * sizeof(float));
  batch_fuse_input_2->CopyDataToTensor(input_data_2.data(),
                                     input_data_2.size() * sizeof(float));

  batch_fuse_graph->Run();
  std::vector<float> out_data;
  auto batch_fuse_out_shape = batch_fuse_output->GetShape();
  out_data.resize(24);
  batch_fuse_output->CopyDataFromTensor(out_data.data());
  std::vector<float> golden = {0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8};
  EXPECT_EQ(out_data, golden);
}