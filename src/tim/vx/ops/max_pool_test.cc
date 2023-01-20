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

TEST(MAX, shape_32_3_1_fp32_kernel_2_stride_1) {
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
             1.764052391052246,
            0.978738009929657,
            2.2408931255340576,
            2.2408931255340576,
            1.8675580024719238,
            0.9500884413719177,
            0.9500884413719177,
            -0.10321885347366333,
            0.4105985164642334,
            0.4105985164642334,
            1.4542734622955322,
            1.4542734622955322,
            0.7610377073287964,
            0.44386324286460876,
            0.44386324286460876,
            1.4940791130065918,
            1.4940791130065918,
            0.3130677044391632,
            0.3130677044391632,
            -0.8540957570075989,
            0.653618574142456,
            0.8644362092018127,
            0.8644362092018127,
            2.269754648208618,
            2.269754648208618,
            0.04575851559638977,
            0.04575851559638977,
            1.5327792167663574,
            1.5327792167663574,
            1.4693588018417358,
            0.37816253304481506,

            -0.8877857327461243,
            -0.34791216254234314,
            0.15634897351264954,
            1.2302906513214111,
            1.2302906513214111,
            1.202379822731018,
            -0.302302747964859,
            -0.302302747964859,
            -1.0485529899597168,
            -1.420017957687378,
            1.950775384902954,
            1.950775384902954,
            -0.4380742907524109,
            -0.4380742907524109,
            0.7774903774261475,
            0.7774903774261475,
            -0.21274028718471527,
            -0.21274028718471527,
            0.38690251111984253,
            0.38690251111984253,
            -0.5108051300048828,
            -0.02818222902715206,
            0.4283318817615509,
            0.4283318817615509,
            0.30247190594673157,
            0.30247190594673157,
            -0.3627411723136902,
            -0.3627411723136902,
            -0.35955315828323364,
            -0.35955315828323364,
            -0.8131462931632996,

             0.17742614448070526,
            -0.4017809331417084,
            0.46278226375579834,
            0.46278226375579834,
            0.05194539576768875,
            0.7290905714035034,
            0.7290905714035034,
            1.1394007205963135,
            1.1394007205963135,
            0.4023416340351105,
            0.4023416340351105,
            -0.6848101019859314,
            -0.5788496732711792,
            -0.3115525245666504,
            0.056165341287851334,
            0.056165341287851334,
            0.9008265137672424,
            0.9008265137672424,
            0.4656624495983124,
            1.4882521629333496,
            1.895889163017273,
            1.895889163017273,
            1.1787796020507812,
            -0.1799248307943344,
            1.0544517040252686,
            1.0544517040252686,
            1.222445011138916,
            1.222445011138916,
            0.9766390323638916,
            0.9766390323638916,
            0.7065731883049011
        };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size()*4));
    uint32_t ksize = 2;
    uint32_t stride = 1;
    auto op = graph->CreateOperation<tim::vx::ops::Pool1d>(tim::vx::PoolType::MAX,
        tim::vx::PadType::VALID, ksize, stride);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_EQ(golden, output);
}

TEST(MAX, shape_6_6_1_1_fp32_kernel_3_stride_2) {
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType in_shape({6, 6, 1, 1});
  tim::vx::ShapeType out_shape({3, 3, 1, 1});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, in_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, out_shape,
                                  tim::vx::TensorAttribute::OUTPUT);

  auto input_tensor = graph->CreateTensor(input_spec);
  auto output_tensor = graph->CreateTensor(output_spec);

  std::vector<float> in_data = {
      -1,  -2,  -3,  -4,  -5,  -5,
      -6,  -7,  -8,  -9,  -10, -10,
      -11, -12, -13, -14, -15, -15,
      -16, -17, -18, -19, -20, -20,
      -21, -22, -23, -24, -25, -20,
      -21, -22, -23, -24, -25, -20,
  };
  std::vector<float> golden = {
      -1, -3, -5,
      -11, -13, -15,
      -21, -23, -20,
  };

  EXPECT_TRUE(
      input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * 4));
  std::array<uint32_t, 4> pad = {0, 1, 0, 1};
  std::array<uint32_t, 2> ksize = {3, 3};
  std::array<uint32_t, 2> stride = {2, 2};
  auto round_type = tim::vx::RoundType::CEILING;
  auto op = graph->CreateOperation<tim::vx::ops::Pool2d>(
      tim::vx::PoolType::MAX, pad, ksize, stride, round_type);
  (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

  EXPECT_TRUE(graph->Compile());
  EXPECT_TRUE(graph->Run());

  std::vector<float> output(golden.size());
  EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
  EXPECT_EQ(golden, output);
}