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
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/pool1d.h"
#include <iostream>
#include "gtest/gtest.h"
#include "test_utils.h"

TEST(AVG, shape_32_3_1_fp32_kernel_2_stride_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({32, 3, 1});
    tim::vx::ShapeType out_shape({31, 3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
            1.764052391052246,
            0.40015721321105957,
            0.978738009929657,
            2.2408931255340576,
            1.8675580024719238,
            -0.9772778749465942,
            0.9500884413719177,
            -0.15135720372200012,
            -0.10321885347366333,
            0.4105985164642334,
            0.14404356479644775,
            1.4542734622955322,
            0.7610377073287964,
            0.12167501449584961,
            0.44386324286460876,
            0.3336743414402008,
            1.4940791130065918,
            -0.2051582634449005,
            0.3130677044391632,
            -0.8540957570075989,
            -2.5529897212982178,
            0.653618574142456,
            0.8644362092018127,
            -0.7421650290489197,
            2.269754648208618,
            -1.4543657302856445,
            0.04575851559638977,
            -0.18718385696411133,
            1.5327792167663574,
            1.4693588018417358,
            0.154947429895401,
            0.37816253304481506,

            -0.8877857327461243,
            -1.980796456336975,
            -0.34791216254234314,
            0.15634897351264954,
            1.2302906513214111,
            1.202379822731018,
            -0.38732680678367615,
            -0.302302747964859,
            -1.0485529899597168,
            -1.420017957687378,
            -1.7062702178955078,
            1.950775384902954,
            -0.5096521973609924,
            -0.4380742907524109,
            -1.2527953386306763,
            0.7774903774261475,
            -1.6138978004455566,
            -0.21274028718471527,
            -0.8954665660858154,
            0.38690251111984253,
            -0.5108051300048828,
            -1.18063223361969,
            -0.02818222902715206,
            0.4283318817615509,
            0.06651721894741058,
            0.30247190594673157,
            -0.6343221068382263,
            -0.3627411723136902,
            -0.6724604368209839,
            -0.35955315828323364,
            -0.8131462931632996,
            -1.7262825965881348,

             0.17742614448070526,
            -0.4017809331417084,
            -1.630198359489441,
            0.46278226375579834,
            -0.9072983860969543,
            0.05194539576768875,
            0.7290905714035034,
            0.12898291647434235,
            1.1394007205963135,
            -1.234825849533081,
            0.4023416340351105,
            -0.6848101019859314,
            -0.8707971572875977,
            -0.5788496732711792,
            -0.3115525245666504,
            0.056165341287851334,
            -1.1651498079299927,
            0.9008265137672424,
            0.4656624495983124,
            -1.5362436771392822,
            1.4882521629333496,
            1.895889163017273,
            1.1787796020507812,
            -0.1799248307943344,
            -1.0707526206970215,
            1.0544517040252686,
            -0.4031769335269928,
            1.222445011138916,
            0.2082749754190445,
            0.9766390323638916,
            0.3563663959503174,
            0.7065731883049011
        };
    std::vector<float> golden = {
            1.0821048021316528,
            0.6894476413726807,
            1.6098155975341797,
            2.054225444793701,
            0.4451400637626648,
            -0.013594716787338257,
            0.3993656039237976,
            -0.12728802859783173,
            0.15368983149528503,
            0.2773210406303406,
            0.79915851354599,
            1.1076555252075195,
            0.441356360912323,
            0.2827691435813904,
            0.3887687921524048,
            0.9138767123222351,
            0.6444604396820068,
            0.05395472049713135,
            -0.27051401138305664,
            -1.703542709350586,
            -0.9496855735778809,
            0.759027361869812,
            0.06113559007644653,
            0.7637947797775269,
            0.4076944589614868,
            -0.7043036222457886,
            -0.07071267068386078,
            0.672797679901123,
            1.5010690689086914,
            0.8121531009674072,
            0.26655498147010803,

            -1.434291124343872,
            -1.1643543243408203,
            -0.0957815945148468,
            0.6933197975158691,
            1.2163352966308594,
            0.40752649307250977,
            -0.3448147773742676,
            -0.6754278540611267,
            -1.2342854738235474,
            -1.5631440877914429,
            0.12225258350372314,
            0.7205616235733032,
            -0.47386324405670166,
            -0.8454347848892212,
            -0.2376524806022644,
            -0.4182037115097046,
            -0.9133190512657166,
            -0.554103434085846,
            -0.25428202748298645,
            -0.06195130944252014,
            -0.8457186818122864,
            -0.6044072508811951,
            0.20007482171058655,
            0.24742454290390015,
            0.18449455499649048,
            -0.16592510044574738,
            -0.49853163957595825,
            -0.5176007747650146,
            -0.5160068273544312,
            -0.5863497257232666,
            -1.2697144746780396,

            -0.11217739433050156,
            -1.0159896612167358,
            -0.5837080478668213,
            -0.222258061170578,
            -0.4276764988899231,
            0.3905179798603058,
            0.4290367364883423,
            0.6341918110847473,
            -0.04771256446838379,
            -0.4162421226501465,
            -0.14123423397541046,
            -0.7778036594390869,
            -0.7248234152793884,
            -0.4452010989189148,
            -0.12769359350204468,
            -0.5544922351837158,
            -0.13216164708137512,
            0.6832444667816162,
            -0.5352905988693237,
            -0.02399575710296631,
            1.692070722579956,
            1.5373344421386719,
            0.4994273781776428,
            -0.6253387331962585,
            -0.008150458335876465,
            0.3256374001502991,
            0.4096340537071228,
            0.7153599858283997,
            0.5924569964408875,
            0.6665027141571045,
            0.5314698219299316
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    uint32_t ksize = 2;
    uint32_t stride = 1;
    auto op = graph->CreateOperation<tim::vx::ops::Pool1d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AVG, shape_3_3_1_2_fp32_kernel_2_stride_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 3, 1, 2});
    tim::vx::ShapeType out_shape({2, 2, 1, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,

        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        };
    std::vector<float> golden = {
        3, 4,
        6, 7,

        3, 4,
        6, 7,
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {1, 1};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AVG, shape_3_3_1_1_fp32_kernel_2_stride_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 3, 1, 1});
    tim::vx::ShapeType out_shape({2, 2, 1, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
        };
    std::vector<float> golden = {
        3, 4,
        6, 7
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {1, 1};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(AVG, shape_3_3_1_1_uint8_kernel_2_stride_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    std::pair<float, int32_t> scalesAndZp;
    scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({3, 3, 1, 1});
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
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
        };
    std::vector<float> golden_float = {
        3, 4,
        6, 7
        };

    std::vector<uint8_t> in_data =
      Quantize<uint8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 2> ksize = {2, 2};
    std::array<uint32_t, 2> stride = {1, 1};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, (uint8_t)1));
}

TEST(AVG, shape_60_52_3_5_fp32_kernel_35_stride_5) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({60, 52, 3, 5});
    tim::vx::ShapeType out_shape({18, 16, 3, 5});
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
                    in_data.push_back(m);
                }
            }
        }
    }
    std::vector<float> slince = {
        0.0408163, 0.183673, 0.428571, 0.77551, 1.22449, 1.77551, 2.42857, 3.14286, 3.85714, 4.57143, 5.28571, 6, 5.44898, 4.79592, 4.04082, 3.18367, 2.22449, 1.16327,
        0.0816327, 0.367347, 0.857143, 1.55102, 2.44898, 3.55102, 4.85714, 6.28571, 7.71429, 9.14286, 10.5714, 12, 10.898, 9.59184, 8.08163, 6.36735, 4.44898, 2.32653,
        0.122449, 0.55102, 1.28571, 2.32653, 3.67347, 5.32653, 7.28571, 9.42857, 11.5714, 13.7143, 15.8571, 18, 16.3469, 14.3878, 12.1224, 9.55102, 6.67347, 3.4898,
        0.163265, 0.734694, 1.71429, 3.10204, 4.89796, 7.10204, 9.71429, 12.5714, 15.4286, 18.2857, 21.1429, 24, 21.7959, 19.1837, 16.1633, 12.7347, 8.89796, 4.65306,
        0.204082, 0.918367, 2.14286, 3.87755, 6.12245, 8.87755, 12.1429, 15.7143, 19.2857, 22.8571, 26.4286, 30, 27.2449, 23.9796, 20.2041, 15.9184, 11.1224, 5.81633,
        0.244898, 1.10204, 2.57143, 4.65306, 7.34694, 10.6531, 14.5714, 18.8571, 23.1429, 27.4286, 31.7143, 36, 32.6939, 28.7755, 24.2449, 19.102, 13.3469, 6.97959,
        0.285714, 1.28571, 3, 5.42857, 8.57143, 12.4286, 17, 22, 27, 32, 37, 42, 38.1429, 33.5714, 28.2857, 22.2857, 15.5714, 8.14286,
        0.285714, 1.28571, 3, 5.42857, 8.57143, 12.4286, 17, 22, 27, 32, 37, 42, 38.1429, 33.5714, 28.2857, 22.2857, 15.5714, 8.14286,
        0.285714, 1.28571, 3, 5.42857, 8.57143, 12.4286, 17, 22, 27, 32, 37, 42, 38.1429, 33.5714, 28.2857, 22.2857, 15.5714, 8.14286,
        0.285714, 1.28571, 3, 5.42857, 8.57143, 12.4286, 17, 22, 27, 32, 37, 42, 38.1429, 33.5714, 28.2857, 22.2857, 15.5714, 8.14286,
        0.261224, 1.17551, 2.74286, 4.96327, 7.83673, 11.3633, 15.5429, 20.1143, 24.6857, 29.2571, 33.8286, 38.4, 34.8735, 30.6939, 25.8612, 20.3755, 14.2367, 7.4449,
        0.220408, 0.991837, 2.31429, 4.18776, 6.61225, 9.58776, 13.1143, 16.9714, 20.8286, 24.6857, 28.5429, 32.4, 29.4245, 25.898, 21.8204, 17.1918, 12.0122, 6.28163,
        0.179592, 0.808163, 1.88571, 3.41224, 5.38775, 7.81224, 10.6857, 13.8286, 16.9714, 20.1143, 23.2571, 26.4, 23.9755, 21.102, 17.7796, 14.0082, 9.78776, 5.11837,
        0.138776, 0.62449, 1.45714, 2.63673, 4.16327, 6.03673, 8.25714, 10.6857, 13.1143, 15.5429, 17.9714, 20.4, 18.5265, 16.3061, 13.7388, 10.8245, 7.56327, 3.9551,
        0.0979592, 0.440816, 1.02857, 1.86122, 2.93878, 4.26122, 5.82857, 7.54286, 9.25714, 10.9714, 12.6857, 14.4, 13.0776, 11.5102, 9.69796, 7.64082, 5.33878, 2.79184,
        0.0571429, 0.257143, 0.6, 1.08571, 1.71429, 2.48571, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 7.62857, 6.71429, 5.65714, 4.45714, 3.11429, 1.62857
        };
    std::vector<float> golden;
    for(int i = 0; i < 15; i++){
        std::copy(slince.begin(), slince.end(),std::back_inserter(golden));
    }
    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 4> pad = {30, 30, 30, 30};
    std::array<uint32_t, 2> ksize = {35, 35};
    std::array<uint32_t, 2> stride = {5, 5};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG,
        pad, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    ArraysMatch(golden, output,1e-4f);
}

TEST(AVG_ANDROID, shape_60_52_3_5_fp32_kernel_35_stride_5) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({60, 52, 3, 5});
    tim::vx::ShapeType out_shape({18, 16, 3, 5});
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
                    in_data.push_back(m);
                }
            }
        }
    }
    std::vector<float> golden;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 16; k++){
                golden.push_back(2);
                golden.push_back(4.5);
                golden.push_back(7);
                golden.push_back(9.5);
                golden.push_back(12);
                golden.push_back(14.5);
                golden.push_back(17);
                golden.push_back(22);
                golden.push_back(27);
                golden.push_back(32);
                golden.push_back(37);
                golden.push_back(42);
                golden.push_back(44.5);
                golden.push_back(47);
                golden.push_back(49.5);
                golden.push_back(52);
                golden.push_back(54.5);
                golden.push_back(57);
            }
        }
    }

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 4> pad = {30, 30, 30, 30};
    std::array<uint32_t, 2> ksize = {35, 35};
    std::array<uint32_t, 2> stride = {5, 5};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    ArraysMatch(golden, output, 1e-5f);
}

TEST(AVG_ANDROID, shape_60_52_3_5_fp32_kernel_50_stride_5) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({60, 52, 3, 5});  //WHCN
    tim::vx::ShapeType out_shape({13, 11, 3, 5});

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
        pad, ksize, stride, tim::vx::RoundType::FLOOR, tim::vx::DataLayout::WHCN);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});
    std::vector<float> output(golden.size());

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));
    EXPECT_TRUE(graph->Run());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));

    ArraysMatch(golden, output, 1e-5f);
}

TEST(AVG_ANDROID, shape_60_52_3_5_uint8_kernel_35_stride_5) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    const float InputMin = -128, InputMax = 127, OutputMin = -128, OutputMax = 127;

    std::pair<float, int32_t> scalesAndZp;
    scalesAndZp = QuantizationParams<uint8_t>(InputMin, InputMax);
    std::vector<float> scalesInput = {scalesAndZp.first};   //scale
    std::vector<int32_t> zeroPointsInput = {scalesAndZp.second}; //zero point

    scalesAndZp = QuantizationParams<uint8_t>(OutputMin, OutputMax);
    std::vector<float> scalesOutput = {scalesAndZp.first};
    std::vector<int32_t> zeroPointsOutput = {scalesAndZp.second};

    tim::vx::ShapeType in_shape({60, 52, 3, 5});
    tim::vx::ShapeType out_shape({18, 16, 3, 5});

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

    std::vector<float> in_data_float;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 52; k++){
                for(int m = 0; m < 60; m++){
                    in_data_float.push_back(m);
                }
            }
        }
    }
    std::vector<float> golden_float;
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 16; k++){
                golden_float.push_back(2);
                golden_float.push_back(4.5);
                golden_float.push_back(7);
                golden_float.push_back(9.5);
                golden_float.push_back(12);
                golden_float.push_back(14.5);
                golden_float.push_back(17);
                golden_float.push_back(22);
                golden_float.push_back(27);
                golden_float.push_back(32);
                golden_float.push_back(37);
                golden_float.push_back(42);
                golden_float.push_back(44.5);
                golden_float.push_back(47);
                golden_float.push_back(49.5);
                golden_float.push_back(52);
                golden_float.push_back(54.5);
                golden_float.push_back(57);
            }
        }
    }

    std::vector<uint8_t> in_data =
      Quantize<uint8_t>(in_data_float, scalesInput[0], zeroPointsInput[0]);   //Quantification process
    std::vector<uint8_t> golden =
      Quantize<uint8_t>(golden_float, scalesOutput[0], zeroPointsOutput[0]);

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    std::array<uint32_t, 4> pad = {30, 30, 30, 30};
    std::array<uint32_t, 2> ksize = {35, 35};
    std::array<uint32_t, 2> stride = {5, 5};
    auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(tim::vx::PoolType::AVG_ANDROID,
        pad, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint8_t> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, (uint8_t)1));
}