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
#include "tim/vx/ops/addn.h"
#include "tim/vx/types.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(AddN, DISABLED_shape_2_2_int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({2, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<int32_t> in_data_x = {
        3, 5,
        4, 8 };
    std::vector<int32_t> in_data_y = {
        1, 6,
        2, 9 };
    std::vector<int32_t> golden = {
        4, 11,
        6, 17 };  //correct answer

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::AddN>(2);   //To refer to the AddN function definition to give the parameters
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int32_t> output(4);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AddN, shape_3_1_float32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 1});
    tim::vx::ShapeType out_shape({3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor_x = graph->CreateTensor(input_spec);
    auto input_tensor_y = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_x = {
        3, 5, 7 };
    std::vector<float> in_data_y = {
        1, 6, 2 };
    std::vector<float> golden = {
        4, 11, 9 };

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(in_data_x.data(), in_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(in_data_y.data(), in_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::AddN>(2);
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(3);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AddN, shape_2_2_uint8_Quantized) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({2, 2});
    tim::vx::ShapeType out_shape({2, 2});

    float InputMin = -127, InputMax = 128, OutputMin = -127, OutputMax = 128;

    std::pair<float, int32_t> scalesAndZp;

    scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};


    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);

    tim::vx::TensorSpec input_spec_x(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec input_spec_y(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);

    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor_x = graph->CreateTensor(input_spec_x);
    auto input_tensor_y = graph->CreateTensor(input_spec_y);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_float_data_x = {
        3.1, 5.1,
        4.1, 8 };
    std::vector<float> in_float_data_y = {
        1.1, 6.1,
        2.1, 9 };
    std::vector<float> golden_float = {
        4.2, 11.2,
        6.2, 17 };

    std::vector<uint8_t> input_data_x =
      Quantize<uint8_t>(in_float_data_x, scalesInput[0], zeroPointsInput[0]);//Quantification process
    std::vector<uint8_t> input_data_y =
      Quantize<uint8_t>(in_float_data_y, scalesInput[0], zeroPointsInput[0]);
    std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor_x->CopyDataToTensor(input_data_x.data(), input_data_x.size()*4));
    EXPECT_TRUE(input_tensor_y->CopyDataToTensor(input_data_y.data(), input_data_y.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::AddN>(2);
    (*op).BindInputs({input_tensor_x, input_tensor_y}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(4);

    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, (uint8_t)1));
}