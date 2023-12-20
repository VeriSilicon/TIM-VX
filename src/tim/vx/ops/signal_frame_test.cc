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
#include "tim/vx/ops/signal_frame.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(SignalFrame, shape_10_3_float_step_2_windows_4) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({10, 3});
    tim::vx::ShapeType out_shape({4, 4, 3});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        0.9854245 ,  1.3478903 ,  2.079034  ,  0.5336022 , -0.8521084 ,
        1.4714626 , -1.6673858 ,  1.1760164 ,  0.58944523, -0.38136077,
        0.4713266 , -0.54476035,  0.17260066,  0.4458921 ,  0.07180826,
       -0.5209453 ,  0.67287415, -0.40036386,  1.819254  , -0.83165807,
        0.7842376 , -0.51183605,  0.5516365 , -0.3449794 , -0.4545289 ,
        1.4418068 ,  2.6290808 ,  0.26231438, -0.50589   , -1.903558  ,
        };

    std::vector<float> golden = {
        0.9854245 ,  1.3478903 ,  2.079034  ,  0.5336022 ,
        2.079034  ,  0.5336022 , -0.8521084 ,  1.4714626 ,
       -0.8521084 ,  1.4714626 , -1.6673858 ,  1.1760164 ,
       -1.6673858 ,  1.1760164 ,  0.58944523, -0.38136077,

        0.4713266 , -0.54476035,  0.17260066,  0.4458921 ,
        0.17260066,  0.4458921 ,  0.07180826, -0.5209453 ,
        0.07180826, -0.5209453 ,  0.67287415, -0.40036386,
        0.67287415, -0.40036386,  1.819254  , -0.83165807,

        0.7842376 , -0.51183605,  0.5516365 , -0.3449794 ,
        0.5516365 , -0.3449794 , -0.4545289 ,  1.4418068 ,
       -0.4545289 ,  1.4418068 ,  2.6290808 ,  0.26231438,
        2.6290808 ,  0.26231438, -0.50589   , -1.903558  ,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::SignalFrame>(4, 2, 0, 0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size() * sizeof(float));
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(ArraysMatch(golden, output, 1e-5f));
}