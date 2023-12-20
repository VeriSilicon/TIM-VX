#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"
TEST(AVG_ANDROID, layout_infer_) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 60, 52, 5}); //CWHN
    tim::vx::ShapeType out_shape({3, 13, 11, 5});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    std::vector<float> in_data;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 52; k++){
                for(int m = 0; m < 60; m++){
                    in_data.push_back(1);
                }
            }
        }
    }
    std::vector<float> golden;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 11; k++){
                for(int k = 0; k < 13; k++){
                    golden.push_back(1);
                }
            }
        }
    }
    std::array<uint32_t, 4> pad = {50, 50, 50, 50};
    std::array<uint32_t, 2> ksize = {100, 100};
    std::array<uint32_t, 2> stride = {5, 5};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride, tim::vx::RoundType::FLOOR, tim::vx::DataLayout::CWHN);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});
    std::vector<float> output(golden.size());

    auto transform = tim::transform::LayoutInference(graph, ctx);
    auto infer_graph = transform.first;
    auto graph_io_map = transform.second;
    auto infer_input = graph_io_map[graph->InputsTensor()[0]];
    auto infer_output = graph_io_map[graph->OutputsTensor()[0]];

    EXPECT_TRUE(infer_graph->Compile());
    EXPECT_TRUE(infer_input->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(infer_graph->Run());
    EXPECT_TRUE(infer_output->CopyDataFromTensor(output.data()));

    EXPECT_EQ(golden, output);
}