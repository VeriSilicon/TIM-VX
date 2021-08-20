/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/pool2d.h"
#include "src/tim/vx/test_utils.h"
#include "gtest/gtest.h"

TEST(Avg, shape_4_4_1_1_fp32_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden = {
        3.5, 5.5,
        11.5, 13.5 };
    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Avg, shape_4_4_1_1_uint8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;    //Quantification range

    scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5 };

    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<uint8_t> in_data=
      Quantize<uint8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Avg, shape_4_4_1_1_int8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5 };

    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<int8_t> in_data=
      Quantize<int8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(Avg, shape_4_4_1_2_int8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 2});
    tim::vx::ShapeType out_shape({2, 2, 1, 2});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,

        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5,

        3.5, 5.5,
        11.5, 13.5
        };
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<int8_t> in_data=
      Quantize<int8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AvgAndroid, shape_4_4_1_1_fp32_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(out_spec);

    std::vector<float> in_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden = {
        3.5, 5.5,
        11.5, 13.5 };
    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AvgAndroid, shape_4_4_1_1_uint8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5 };
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<uint8_t> in_data=
      Quantize<uint8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AvgAndroid, shape_4_4_1_1_int8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5 };
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<int8_t> in_data=
      Quantize<int8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AvgAndroid, shape_4_4_1_2_int8_kernel_2_stride_2){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    std::pair<float, int32_t> scalesAndZp;
    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    scalesAndZp = QuantizationParams<int8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<int8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({4, 4, 1, 2});
    tim::vx::ShapeType out_shape({2, 2, 1, 2});

    tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
                                   scalesInput, zeroPointsInput);
    tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
                                    scalesOutput, zeroPointsOutput);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::INT8, in_shape,
                                 tim::vx::TensorAttribute::INPUT, quantInput);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::INT8, out_shape,
                                tim::vx::TensorAttribute::OUTPUT, quantOutput);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data_float = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,

        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
        };
    std::vector<float> golden_float = {
        3.5, 5.5,
        11.5, 13.5,

        3.5, 5.5,
        11.5, 13.5
        };
    std::array<uint32_t, 4> pad = {0, 0, 0, 0};
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {2, 2};

    std::vector<int8_t> in_data=
      Quantize<int8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<int8_t> golden =
      Quantize<int8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInput(input_tensor).BindOutput(output_tensor);

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());
    std::vector<int8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}
