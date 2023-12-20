#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/transpose.h"

#include "gtest/gtest.h"

TEST(Transpose, shape_1_3_float_3_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({1,3});
    tim::vx::ShapeType output_shape({3,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1,2,3
        };
    std::vector<float> golden = {
        1,2,3
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(int32_t)));

    std::vector<u_int32_t> perm = {1, 0};
    auto op = graph->CreateOperation<tim::vx::ops::Transpose>(perm);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Transpose, shape_1_3_int32_3_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape({1,3});
    tim::vx::ShapeType output_shape({3,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> in_data = {
        1,2,3
        };
    std::vector<int32_t> golden = {
        1,2,3
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));

    std::vector<u_int32_t> perm = {1, 0};
    auto op = graph->CreateOperation<tim::vx::ops::Transpose>(perm);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<int32_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}