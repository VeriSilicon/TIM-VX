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
#include "tim/vx/ops/svdf.h"
#include "test_utils.h"

#include "gtest/gtest.h"

TEST(Svdf, shape_3_2_10_1_4_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t input_size = 3, batches = 2, memory_size = 10, rank = 1, num_units = 4, spectrogram_length = num_units;

    tim::vx::ShapeType input_shape({input_size, batches});
    tim::vx::ShapeType state_in_shape({rank * num_units * memory_size, batches});
    tim::vx::ShapeType weights_feature_shape({input_size, rank * num_units});
    tim::vx::ShapeType weights_time_shape({memory_size, rank * num_units});
    tim::vx::ShapeType output_shape({num_units, batches});
    tim::vx::ShapeType state_out_shape({rank * num_units * memory_size, batches});

    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32,
        state_in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_feature_spec(tim::vx::DataType::FLOAT32,
        weights_feature_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_time_spec(tim::vx::DataType::FLOAT32,
        weights_time_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec state_out_spec(tim::vx::DataType::FLOAT32,
        state_out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);
    auto weights_feature_tensor = graph->CreateTensor(weights_feature_spec);
    auto weights_time_tensor = graph->CreateTensor(weights_time_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto state_out_tensor = graph->CreateTensor(state_out_spec);

    std::vector<float> in_data = {
        0.12609188,  -0.46347019, -0.89598465,
        0.35867718,  0.36897406,  0.73463392,
        };

    std::vector<float> state_in;
    for(uint32_t m = 0; m < rank * num_units * memory_size * batches; m++)
        state_in.push_back(0);

    std::vector<float> weights_feature = {
        -0.31930989, -0.36118156, 0.0079667,
         0.37613347,  0.22197971, 0.12416199,
         0.27901134,  0.27557442, 0.3905206,
        -0.36137494, -0.06634006, -0.10640851};

    std::vector<float> weights_time = {
        -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

    std::vector<float> golden = {
         0.014899,   -0.0517661,  -0.143725,   -0.00271883,
        -0.03004015, 0.09565311,  0.1587342,   0.00784263,
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(state_in_tensor->CopyDataToTensor(
        state_in.data(), state_in.size() * sizeof(float)));
    EXPECT_TRUE(weights_feature_tensor->CopyDataToTensor(
        weights_feature.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weights_time_tensor->CopyDataToTensor(
        weights_time.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Svdf>(rank, num_units, spectrogram_length);
    (*op).BindInputs({input_tensor, state_in_tensor, weights_feature_tensor, weights_time_tensor})
         .BindOutputs({output_tensor, state_out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    // EXPECT_EQ(golden, output);
    EXPECT_TRUE(ArraysMatch(golden, output,1e-5f));
}

TEST(Svdf, shape_3_2_10_2_4_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t input_size = 3, batches = 2, memory_size = 10, rank = 2, num_units = 4, spectrogram_length = num_units;

    tim::vx::ShapeType input_shape({input_size, batches});
    tim::vx::ShapeType state_in_shape({rank * num_units * memory_size, batches});
    tim::vx::ShapeType weights_feature_shape({input_size, rank * num_units});
    tim::vx::ShapeType weights_time_shape({memory_size, rank * num_units});
    tim::vx::ShapeType output_shape({num_units, batches});
    tim::vx::ShapeType state_out_shape({rank * num_units * memory_size, batches});

    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32,
        state_in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_feature_spec(tim::vx::DataType::FLOAT32,
        weights_feature_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_time_spec(tim::vx::DataType::FLOAT32,
        weights_time_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec state_out_spec(tim::vx::DataType::FLOAT32,
        state_out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);
    auto weights_feature_tensor = graph->CreateTensor(weights_feature_spec);
    auto weights_time_tensor = graph->CreateTensor(weights_time_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto state_out_tensor = graph->CreateTensor(state_out_spec);

    std::vector<float> in_data = {
        0.12609188,  -0.46347019, -0.89598465,
        0.35867718,  0.36897406,  0.73463392,};

    std::vector<float> state_in;
    for(uint32_t m = 0; m < rank * num_units * memory_size * batches; m++)
        state_in.push_back(0);

    std::vector<float> weights_feature = {
        -0.31930989, 0.0079667,   0.39296314,
        0.37613347,  0.12416199,  0.15785322,
        0.27901134,  0.3905206,   0.21931258,
        -0.36137494, -0.10640851, 0.31053296,
        -0.36118156, -0.0976817,  -0.36916667,
        0.22197971,   0.15294972,  0.38031587,
        0.27557442,  0.39635518,  -0.21580373,
        -0.06634006, -0.02702999,  0.27072677
        };

    std::vector<float> weights_time = {
        -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763,
       };

    std::vector<float> golden = {
         -0.09623547, -0.10193135, 0.11083051,  -0.0347917,
        0.1141196,   0.12965347,  -0.12652366, 0.01007236,
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(state_in_tensor->CopyDataToTensor(
        state_in.data(), state_in.size() * sizeof(float)));
    EXPECT_TRUE(weights_feature_tensor->CopyDataToTensor(
        weights_feature.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weights_time_tensor->CopyDataToTensor(
        weights_time.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Svdf>(rank, num_units, spectrogram_length);
    (*op).BindInputs({input_tensor, state_in_tensor, weights_feature_tensor, weights_time_tensor})
         .BindOutputs({output_tensor, state_out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    // EXPECT_EQ(golden, output);
    EXPECT_TRUE(ArraysMatch(golden, output,1e-5f));
}

TEST(Svdf, shape_3_2_10_3_4_float) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    uint32_t input_size = 3, batches = 2, memory_size = 10, rank = 3, num_units = 4, spectrogram_length = num_units;

    tim::vx::ShapeType input_shape({input_size, batches});
    tim::vx::ShapeType state_in_shape({rank * num_units * memory_size, batches});
    tim::vx::ShapeType weights_feature_shape({input_size, rank * num_units});
    tim::vx::ShapeType weights_time_shape({memory_size, rank * num_units});
    tim::vx::ShapeType output_shape({num_units, batches});
    tim::vx::ShapeType state_out_shape({rank * num_units * memory_size, batches});

    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
        input_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec state_in_spec(tim::vx::DataType::FLOAT32,
        state_in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_feature_spec(tim::vx::DataType::FLOAT32,
        weights_feature_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec weights_time_spec(tim::vx::DataType::FLOAT32,
        weights_time_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
        output_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec state_out_spec(tim::vx::DataType::FLOAT32,
        state_out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto state_in_tensor = graph->CreateTensor(state_in_spec);
    auto weights_feature_tensor = graph->CreateTensor(weights_feature_spec);
    auto weights_time_tensor = graph->CreateTensor(weights_time_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto state_out_tensor = graph->CreateTensor(state_out_spec);

    std::vector<float> in_data = {
        0.12609188,  -0.46347019, -0.89598465,
        0.35867718,  0.36897406,  0.73463392,};

    std::vector<float> state_in;
    for(uint32_t m = 0; m < rank * num_units * memory_size * batches; m++)
        state_in.push_back(0);

    std::vector<float> weights_feature = {
        -0.31930989, 0.0079667,   0.39296314,
        0.37613347,  0.12416199,  0.15785322,
        0.27901134,  0.3905206,   0.21931258,
        -0.36137494, -0.10640851, 0.31053296,
        -0.36118156, -0.0976817,  -0.36916667,
        0.22197971,   0.15294972,  0.38031587,
        0.27557442,  0.39635518,  -0.21580373,
        -0.06634006, -0.02702999,  0.27072677,
        -0.31930989, -0.36118156, 0.0079667,
         0.37613347,  0.22197971, 0.12416199,
         0.27901134,  0.27557442, 0.3905206,
        -0.36137494, -0.06634006, -0.10640851
        };

    std::vector<float> weights_time = {
        -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
       0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,

       -0.26816407, -0.19905911, 0.29211238,  0.31264046,  -0.28664589,
       0.05698794,  0.11613581,  0.14078894,  0.02187902,  -0.21781836,

       -0.15567942, 0.08693647,  -0.38256618, 0.36580828,  -0.22922277,
       -0.0226903,  0.12878349,  -0.28122205, -0.10850525, -0.11955214,

       0.27179423,  -0.04710215, 0.31069002,  0.22672787,  0.09580326,
       0.08682203,  0.1258215,   0.1851041,   0.29228821,  0.12366763,

       0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
       -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

       -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
       0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

       -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
       -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657,

       -0.14884081, 0.19931212,  -0.36002168, 0.34663299,  -0.11405486,
       0.12672701,  0.39463779,  -0.07886535, -0.06384811, 0.08249187,
       };

    std::vector<float> golden = {
        -0.207435, 0.120099, 0.00247115, -0.0325705,
        0.245774,  -0.128524, -0.065059, 0.0644025,
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(state_in_tensor->CopyDataToTensor(
        state_in.data(), state_in.size() * sizeof(float)));
    EXPECT_TRUE(weights_feature_tensor->CopyDataToTensor(
        weights_feature.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(weights_time_tensor->CopyDataToTensor(
        weights_time.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Svdf>(rank, num_units, spectrogram_length);
    (*op).BindInputs({input_tensor, state_in_tensor, weights_feature_tensor, weights_time_tensor})
         .BindOutputs({output_tensor, state_out_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    // EXPECT_EQ(golden, output);
    EXPECT_TRUE(ArraysMatch(golden, output,1e-5f));
}

// TEST(Svdf, shape_3_2_10_1_4_uint8) {
//     auto ctx = tim::vx::Context::Create();
//     auto graph = ctx->CreateGraph();

//     const float InputMin = -127, InputMax = 128, OutputMin = -127, OutputMax = 128;
//     uint32_t input_size = 3, batches = 2, memory_size = 10, rank = 1, num_units = 4, spectrogram_length = num_units;

//     std::pair<float, int32_t> scalesAndZp;
//     scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
//     std::vector<float> scalesInput = {scalesAndZp.first};   //scale
//     std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point
//     scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
//     std::vector<float> scalesOutput = {scalesAndZp.first};
//     std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

//     tim::vx::ShapeType input_shape({input_size, batches});
//     tim::vx::ShapeType state_in_shape({rank * num_units * memory_size, batches});
//     tim::vx::ShapeType weights_feature_shape({input_size, rank * num_units});
//     tim::vx::ShapeType weights_time_shape({memory_size, rank * num_units});
//     tim::vx::ShapeType output_shape({num_units, batches});
//     tim::vx::ShapeType state_out_shape({rank * num_units * memory_size, batches});

//     tim::vx::Quantization quantInput(tim::vx::QuantType::ASYMMETRIC, 1,
//                                    scalesInput, zeroPointsInput);
//     tim::vx::Quantization quantOutput(tim::vx::QuantType::ASYMMETRIC, 1,
//                                     scalesOutput, zeroPointsOutput);

//     tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8,
//         input_shape, tim::vx::TensorAttribute::INPUT, quantInput);
//     tim::vx::TensorSpec state_in_spec(tim::vx::DataType::UINT8,
//         state_in_shape, tim::vx::TensorAttribute::INPUT, quantInput);
//     tim::vx::TensorSpec weights_feature_spec(tim::vx::DataType::UINT8,
//         weights_feature_shape, tim::vx::TensorAttribute::INPUT, quantInput);
//     tim::vx::TensorSpec weights_time_spec(tim::vx::DataType::UINT8,
//         weights_time_shape, tim::vx::TensorAttribute::INPUT, quantInput);
//     tim::vx::TensorSpec output_spec(tim::vx::DataType::UINT8,
//         output_shape, tim::vx::TensorAttribute::OUTPUT, quantOutput);
//     tim::vx::TensorSpec state_out_spec(tim::vx::DataType::UINT8,
//         state_out_shape, tim::vx::TensorAttribute::OUTPUT, quantOutput);

//     auto input_tensor = graph->CreateTensor(input_spec);
//     auto state_in_tensor = graph->CreateTensor(state_in_spec);
//     auto weights_feature_tensor = graph->CreateTensor(weights_feature_spec);
//     auto weights_time_tensor = graph->CreateTensor(weights_time_spec);
//     auto output_tensor = graph->CreateTensor(output_spec);
//     auto state_out_tensor = graph->CreateTensor(state_out_spec);

//     std::vector<float> in_data_float = {
//         0.12609188,  -0.46347019, -0.89598465,
//         0.35867718,  0.36897406,  0.73463392,
//         };

//     std::vector<uint8_t> state_in_uint8;
//     for(uint32_t m = 0; m < rank * num_units * memory_size * batches; m++)
//         state_in_uint8.push_back(0);

//     std::vector<float> weights_feature_float = {
//         -0.31930989, -0.36118156, 0.0079667,
//          0.37613347,  0.22197971, 0.12416199,
//          0.27901134,  0.27557442, 0.3905206,
//         -0.36137494, -0.06634006, -0.10640851};

//     std::vector<float> weights_time_float = {
//         -0.31930989, 0.37613347,  0.27901134,  -0.36137494, -0.36118156,
//        0.22197971,  0.27557442,  -0.06634006, 0.0079667,   0.12416199,

//        0.3905206,   -0.10640851, -0.0976817,  0.15294972,  0.39635518,
//        -0.02702999, 0.39296314,  0.15785322,  0.21931258,  0.31053296,

//        -0.36916667, 0.38031587,  -0.21580373, 0.27072677,  0.23622236,
//        0.34936687,  0.18174365,  0.35907319,  -0.17493086, 0.324846,

//        -0.10781813, 0.27201805,  0.14324132,  -0.23681851, -0.27115166,
//        -0.01580888, -0.14943552, 0.15465137,  0.09784451,  -0.0337657};

//     std::vector<float> golden_float = {
//          0.014899,   -0.0517661,  -0.143725,   -0.00271883,
//         -0.03004015, 0.09565311,  0.1587342,   0.00784263,
//         };

//     std::vector<uint8_t> in_data_uint8 =
//       Quantize<uint8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
//     std::vector<uint8_t> weights_feature_uint8 =
//       Quantize<uint8_t>(weights_feature_float, scalesInput[0], zeroPointsInput[0]);
//     std::vector<uint8_t> weights_time_uint8 =
//       Quantize<uint8_t>(weights_time_float, scalesInput[0], zeroPointsInput[0]);
//     std::vector<uint8_t> golden_uint8 =
//       Quantize<uint8_t>(golden_float, scalesInput[0], zeroPointsInput[0]);

//     EXPECT_TRUE(input_tensor->CopyDataToTensor(
//         in_data_uint8.data(), in_data_uint8.size() * sizeof(uint8_t)));
//     EXPECT_TRUE(state_in_tensor->CopyDataToTensor(
//         state_in_uint8.data(), state_in_uint8.size() * sizeof(uint8_t)));
//     EXPECT_TRUE(weights_feature_tensor->CopyDataToTensor(
//         weights_feature_uint8.data(), weights_feature_uint8.size() * sizeof(uint8_t)));
//     EXPECT_TRUE(weights_time_tensor->CopyDataToTensor(
//         weights_time_uint8.data(), weights_time_uint8.size() * sizeof(uint8_t)));

//     auto op = graph->CreateOperation<tim::vx::ops::Svdf>(rank, num_units, spectrogram_length);
//     (*op).BindInputs({input_tensor, state_in_tensor, weights_feature_tensor, weights_time_tensor})
//          .BindOutputs({output_tensor, state_out_tensor});

//     EXPECT_TRUE(graph->Compile());
//     EXPECT_TRUE(graph->Run());

//     std::vector<uint8_t> output_uint8(golden_uint8.size());
//     EXPECT_TRUE(output_tensor->CopyDataFromTensor(output_uint8.data()));
//     EXPECT_EQ(golden_uint8, output_uint8);
//     // EXPECT_TRUE(ArraysMatch(golden, output,1e-5f));
// }
