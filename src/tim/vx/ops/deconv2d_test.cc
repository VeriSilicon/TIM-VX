#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/deconv.h"

#include "gtest/gtest.h"

namespace {

size_t element_count(const tim::vx::ShapeType& shape) {
  size_t sz = 1;
  for (auto d : shape) {
    sz *= d;
  }

  return sz;
}

}  // namespace

TEST(DeConv2d, shape_3_3_2_1_float_depthwise) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape ({3, 3, 2, 1});  //whcn
    tim::vx::ShapeType kernel_shape({3, 3, 2, 1});  //whc1 same as depthwise convolution
    tim::vx::ShapeType output_shape({5, 5, 2, 1});  //whcn

    tim::vx::TensorSpec input_spec  (tim::vx::DataType::FLOAT32, input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec kernel_spec (tim::vx::DataType::FLOAT32, kernel_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec (tim::vx::DataType::FLOAT32, output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto kernel_tensor = graph->CreateTensor(kernel_spec);

    std::vector<float> input_data = {3.0f, 8.0f, 1.0f,
                                     9.0f, 5.0f, 7.0f,
                                     3.0f, 2.0f, 3.0f,

                                     7.0f, 9.0f, 1.0f,
                                     5.0f, 2.0f, 3.0f,
                                     9.0f, 0.0f, 2.0f};
    std::vector<float> kernel_data =
                                    {9.0f, 0.0f, 3.0f,
                                     0.0f, 0.0f, 0.0f,
                                     1.0f, 0.0f, 2.0f,

                                    3.0f, 0.0f, 7.0f,
                                    0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 8.0f,
                                    };

    std::vector<float> output_data(element_count(output_shape));

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()*4));
    EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(), kernel_data.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::DeConv2d>(
        2,
        tim::vx::PadType::SAME,
        std::array<uint32_t, 2>({3, 3}),    /*ksize*/
        std::array<uint32_t, 2>({1, 1}),    /*stride*/
        std::array<uint32_t, 2>({1, 1}),    /*dilation*/
        std::array<uint32_t, 4>({0, 0, 0, 0}), /*pad*/
        2/*group*/);
    (*op).BindInputs({input_tensor, kernel_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));
    std::vector<float> golden = {
                        27.0f, 72.0f, 18.0f, 24.0f, 3.0f,
                        81.0f, 45.0f, 90.0f, 15.0f, 21.0f,
                        30.0f, 26.0f, 43.0f, 22.0f, 11.0f,
                        9.0f, 5.0f, 25.0f, 10.0f, 14.0f,
                        3.0f, 2.0f, 9.0f, 4.0f, 6.0f,

                        21.0f, 27.0f, 52.0f, 63.0f, 7.0f,
                        15.0f, 6.0f, 44.0f, 14.0f, 21.0f,
                        27.0f, 0.0f, 125.0f, 72.0f, 22.0f,
                        0.0f, 0.0f, 40.0f, 16.0f, 24.0f,
                        0.0f, 0.0f, 72.0f, 0.0f, 16.0f};

    EXPECT_EQ(golden, output_data) << "Result mismatch";
}

TEST(DeConv2d, shape_3_3_1_1_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType input_shape ({3, 3, 1, 1});  //whcn
    tim::vx::ShapeType kernel_shape({3, 3, 1, 1});  //whc1 same as depthwise convolution
    tim::vx::ShapeType output_shape({5, 5, 1, 1});  //whcn

    tim::vx::TensorSpec input_spec  (tim::vx::DataType::FLOAT32, input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec kernel_spec (tim::vx::DataType::FLOAT32, kernel_shape, tim::vx::TensorAttribute::CONSTANT);
    tim::vx::TensorSpec output_spec (tim::vx::DataType::FLOAT32, output_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto kernel_tensor = graph->CreateTensor(kernel_spec);

    std::vector<float> input_data = {
        3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 3.0f,
    };

    std::vector<float> kernel_data = {
        9.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f,
    };

    std::vector<float> output_data(element_count(output_shape));

    EXPECT_TRUE(input_tensor->CopyDataToTensor(input_data.data(), input_data.size()*4));
    EXPECT_TRUE(kernel_tensor->CopyDataToTensor(kernel_data.data(), kernel_data.size()*4));

    auto op = graph->CreateOperation<tim::vx::ops::DeConv2d>(
        1,
        tim::vx::PadType::SAME,
        std::array<uint32_t, 2>({3, 3}),    /*ksize*/
        std::array<uint32_t, 2>({1, 1}),    /*stride*/
        std::array<uint32_t, 2>({1, 1}),    /*dilation*/
        std::array<uint32_t, 4>({0, 0, 0, 0}), /*pad*/
        1/*group*/);
    (*op).BindInputs({input_tensor, kernel_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_data.data()));
    std::vector<float> golden = {
        27.0f, 72.0f, 18.0f, 24.0f, 3.0f,  81.0f, 45.0f, 90.0f, 15.0f,
        21.0f, 30.0f, 26.0f, 43.0f, 22.0f, 11.0f, 9.0f,  5.0f,  25.0f,
        10.0f, 14.0f, 3.0f,  2.0f,  9.0f,  4.0f,  6.0f,
    };

    EXPECT_EQ(golden, output_data) << "Result mismatch";
}
