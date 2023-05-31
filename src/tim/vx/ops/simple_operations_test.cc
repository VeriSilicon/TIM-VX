/****************************************************************************
*
*    Copyright (c) 2020-2023 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/ops/simple_operations.h"
#include "test_utils.h"

#include "gtest/gtest.h"
#include <cstdlib>

TEST(Floor, shape_5_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> golden = {-3, -1, 0, 0, std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto add = graph->CreateOperation<tim::vx::ops::Floor>();
    (*add).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Round, shape_15_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({15, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = { 0.1, 0.5, 0.9, 1.2, 1.5,
            1.8, 2.3, 2.5, 2.7, -1.1,
            -1.5, -1.9, -2.2, -2.5, -2.8 };
    std::vector<float> golden = {0., 0., 1., 1., 2.,
            2., 2., 2., 3., -1.,
            -2., -2., -2., -2., -3. };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto add = graph->CreateOperation<tim::vx::ops::Round>();
    (*add).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(15, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Ceil, shape_5_1_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> golden = {-2, 0, 0, 1, std::numeric_limits<float>::infinity() };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto add = graph->CreateOperation<tim::vx::ops::Ceil>();
    (*add).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Cast, shape_5_1_fp32_to_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<int> golden = {-2, 0, 0, 0, std::numeric_limits<int>::max()};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto add = graph->CreateOperation<tim::vx::ops::Cast>();
    (*add).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(DataConvert, quantize_shape_2_3_fp32_to_asym_u8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({2, 3});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape,
                                   tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, io_shape,
                                    tim::vx::TensorAttribute::OUTPUT, quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {0.8458, 0.6214, 0.4666, 0.6065, 0.8895, 0.1535};
    std::vector<uint8_t> golden = {235, 173, 130, 168, 247,  43};

    auto quantize = graph->CreateOperation<tim::vx::ops::DataConvert>();
    (*quantize).BindInput(input_tensor).BindOutput(output_tensor);
    EXPECT_TRUE(graph->Compile());

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(6, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(DataConvert, dequantize_shape_2_3_asym_u8_to_fp32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({2, 3});
    tim::vx::Quantization quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape,
                                   tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, io_shape,
                                    tim::vx::TensorAttribute::INPUT, quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {235, 173, 130, 168, 247,  43};
    std::vector<float> golden = {0.8458, 0.6214, 0.4666, 0.6065, 0.8895, 0.1535};

    auto dequantize = graph->CreateOperation<tim::vx::ops::DataConvert>();
    (*dequantize).BindInput(input_tensor).BindOutput(output_tensor);
    EXPECT_TRUE(graph->Compile());

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(6, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    for (uint32_t idx = 0; idx < output.size(); idx++)
      EXPECT_NEAR(golden[idx], output[idx], 0.01f);
}

TEST(DataConvert, requantize_shape_2_3_asym_u8) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({2, 3});
    tim::vx::Quantization in_quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 0);
    tim::vx::Quantization out_quant(tim::vx::QuantType::ASYMMETRIC, 0.0036, 10);

    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, io_shape,
                                    tim::vx::TensorAttribute::INPUT, in_quant);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, io_shape,
                                   tim::vx::TensorAttribute::OUTPUT, out_quant);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<uint8_t> in_data = {235, 173, 130, 168, 247,  43};
    std::vector<uint8_t> golden = {245, 183, 140, 178, 255,  53};

    auto requantize = graph->CreateOperation<tim::vx::ops::DataConvert>();
    (*requantize).BindInput(input_tensor).BindOutput(output_tensor);
    EXPECT_TRUE(graph->Compile());

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()));
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(6, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Rcp, shape_5_1_fp32) {
   auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({5, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = { -2.5, -0.1, 0, 0.55, std::numeric_limits<float>::infinity() };
    std::vector<float> golden = {-0.4, -10, std::numeric_limits<float>::infinity(), 1.81818, 0.};

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));

    auto add = graph->CreateOperation<tim::vx::ops::Rcp>();
    (*add).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(5, 0);
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}