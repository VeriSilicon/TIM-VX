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
#include "tim/vx/ops/softmax.h"
#include "test_utils.h"
#include "gtest/gtest.h"

TEST(Softmax, shape_3_1_float_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        -1, 0, 1,
    };
    std::vector<float> golden = {
        0.09003057, 0.24472848, 0.66524094
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}

TEST(Softmax, shape_3_4_float_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 4});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
    };
    std::vector<float> golden = {
       0.09003057, 0.24472848, 0.66524094,
       0.09003057, 0.24472848, 0.66524094,
       0.09003057, 0.24472848, 0.66524094,
       0.09003057, 0.24472848, 0.66524094,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}

TEST(Softmax, shape_3_4_float_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 4});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
    };
    std::vector<float> golden = {
       0.25, 0.25, 0.25,
       0.25, 0.25, 0.25,
       0.25, 0.25, 0.25,
       0.25, 0.25, 0.25,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}

TEST(Softmax, shape_3_3_2_float_axis_0) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 3, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,

        2, 3, 5,
        2, 3, 5,
        2, 3, 5,
    };
    std::vector<float> golden = {
       0.09003057, 0.24472848, 0.66524094,
       0.09003057, 0.24472848, 0.66524094,
       0.09003057, 0.24472848, 0.66524094,

       0.04201007, 0.11419519, 0.8437947,
       0.04201007, 0.11419519, 0.8437947,
       0.04201007, 0.11419519, 0.8437947,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,0);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}

TEST(Softmax, shape_3_3_2_float_axis_1) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 3, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,

        2, 3, 5,
        2, 3, 5,
        2, 3, 5,
    };
    std::vector<float> golden = {
       0.33333334, 0.33333334, 0.33333334,
       0.33333334, 0.33333334, 0.33333334,
       0.33333334, 0.33333334, 0.33333334,

       0.33333334, 0.33333334, 0.33333334,
       0.33333334, 0.33333334, 0.33333334,
       0.33333334, 0.33333334, 0.33333334,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,1);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}

TEST(Softmax, shape_3_3_2_float_axis_2) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({3, 3, 2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            io_shape, tim::vx::TensorAttribute::OUTPUT);

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);

    std::vector<float> in_data = {
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,

        2, 3, 5,
        2, 3, 5,
        2, 3, 5,
    };
    std::vector<float> golden = {
       0.26894143, 0.26894143, 0.11920291,
       0.26894143, 0.26894143, 0.11920291,
       0.26894143, 0.26894143, 0.11920291,

       0.7310586 , 0.7310586 , 0.880797,
       0.7310586 , 0.7310586 , 0.880797,
       0.7310586 , 0.7310586 , 0.880797,
    };

    EXPECT_TRUE(input_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(float)));

    auto op = graph->CreateOperation<tim::vx::ops::Softmax>(1,2);
    (*op).BindInputs({input_tensor}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));
    if (!ctx->hasSP())
      EXPECT_EQ(golden, output);
    else
      EXPECT_TRUE(ArraysMatch(golden, output, 1e-3f));
}