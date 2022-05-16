#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/maxpoolwithargmax.h"

#include "gtest/gtest.h"

TEST(MaxpoolWithArgmax, shape_3_3_1_fp32_kernel_2_stride_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 3, 1});
    tim::vx::ShapeType out_shape({2, 2, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9 };
    std::vector<float> values_golden = {
        5, 6,
        8, 9 };
    std::vector<uint8_t> indices_golden = {
        3, 2,
        1, 0 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {1, 1};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(4);
    std::vector<uint8_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}

TEST(MaxpoolWithArgmax, shape_5_5_1_fp32_kernel_3_stride_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({5, 5, 1});
    tim::vx::ShapeType out_shape({2, 2, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec_indices(tim::vx::DataType::UINT8,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec output_spec_values(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor_indices = graph->CreateTensor(output_spec_indices);
    auto output_tensor_values = graph->CreateTensor(output_spec_values);

    std::vector<float> in_data = {
        1, 2, 3, 10, 11,
        4, 5, 6, 0, 1,
        7, 8, 9, 11, -1,
        0, 2, 4, 5, 7,
        20, 1, 1, 0, -1 };
    std::vector<float> values_golden = {
        9, 11,
        20, 11 };
    std::vector<uint8_t> indices_golden = {
        3, 2,
        1, 0 };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {3, 3};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::MaxpoolWithArgmax>(
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor_values, output_tensor_indices});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output_values(4);
    std::vector<uint8_t> output_indices(4);

    EXPECT_TRUE(output_tensor_values->CopyDataFromTensor(output_values.data()));
    EXPECT_TRUE(output_tensor_indices->CopyDataFromTensor(output_indices.data()));
    EXPECT_EQ(values_golden, output_values);
    EXPECT_EQ(indices_golden, output_indices);
}
