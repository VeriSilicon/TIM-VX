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
#include "tim/vx/ops/batchnorm.h"
#include "tim/transform/layout_inference.h"

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(BatchNorm, shape_3_3_2_1_fp32_cwhn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({2, 3, 3, 1});
    tim::vx::ShapeType out_shape({2, 3, 3, 1});
    tim::vx::ShapeType mean_shape({2});
    tim::vx::ShapeType var_shape({2});
    tim::vx::ShapeType gamma_shape({2});
    tim::vx::ShapeType beta_shape({2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec mean_spec(tim::vx::DataType::FLOAT32, mean_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec var_spec(tim::vx::DataType::FLOAT32, var_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec gamma_spec(tim::vx::DataType::FLOAT32, gamma_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec beta_spec(tim::vx::DataType::FLOAT32, beta_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    std::vector<float> in_data = {
        0.59885779, 0.62662862, 0.63011179, 0.82569427, 0.64772359, 0.42895413,
        0.30216458, 0.01351635, 0.32545444, 0.0360674,  0.33967769, 0.18092504,
        0.09479915, 0.52258112, 0.46735646, 0.95689111, 0.51619059, 0.82685718};
    std::vector<float> golden = {
        0.92227477,  0.40612271,  1.09906762,
        1.00176775,  1.19869136,  -0.18535967,
        -0.7560139,  -1.42843423, -0.62427138,
        -1.3609569,  -0.54381545, -0.92751329,
        -1.92900686, 0.09479138,  0.17841823,
        1.39433545,  0.45465564,  1.0052474,
    };

    std::vector<float> mean_data = {
        0.43581513, 0.49090168
    };
    std::vector<float> var_data = {
        0.03025229, 0.11069085
    };
    std::vector<float> gamma_data = {
        1,1
    };
    std::vector<float> beta_data = {
        0, 0
    };

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto mean = graph->CreateTensor(mean_spec, mean_data.data());
    auto var = graph->CreateTensor(var_spec, var_data.data());
    auto gamma = graph->CreateTensor(gamma_spec, gamma_data.data());
    auto beta = graph->CreateTensor(beta_spec, beta_data.data());

    float epsilon = 0.001;
    
    auto op = graph->CreateOperation<tim::vx::ops::BatchNorm>(epsilon, tim::vx::DataLayout::CWHN);
    (*op).BindInputs({input_tensor, mean, var,gamma, beta}).BindOutputs({output_tensor});

    auto final_graph = tim::transform::LayoutInference(graph, ctx);

    EXPECT_TRUE(final_graph.first->Compile());
    final_graph.second[input_tensor]->CopyDataToTensor(
        in_data.data(), in_data.size() * sizeof(float));
    EXPECT_TRUE(final_graph.first->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(final_graph.second[output_tensor]->CopyDataFromTensor(output.data()));

    for (uint32_t idx = 0; idx < golden.size(); idx++) {
      EXPECT_TRUE(std::abs(golden[idx] - output[idx]) < 0.01);
  }
}

TEST(BatchNorm, shape_3_3_2_1_fp32_whcn) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType in_shape({3, 3, 2, 1});
    tim::vx::ShapeType out_shape({3, 3, 2, 1});
    tim::vx::ShapeType mean_shape({2});
    tim::vx::ShapeType var_shape({2});
    tim::vx::ShapeType gamma_shape({2});
    tim::vx::ShapeType beta_shape({2});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec mean_spec(tim::vx::DataType::FLOAT32, mean_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec var_spec(tim::vx::DataType::FLOAT32, var_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec gamma_spec(tim::vx::DataType::FLOAT32, gamma_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec beta_spec(tim::vx::DataType::FLOAT32, beta_shape, tim::vx::CONSTANT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    std::vector<float> in_data = {
        0.598858, 0.630112, 0.647724, 0.302165, 0.325454, 0.339678,
        0.094799, 0.467356, 0.516191, 0.626629, 0.825694, 0.428954,
        0.013516, 0.036067, 0.180925, 0.522581, 0.956891, 0.826857};
    std::vector<float> golden = {
        0.922275,  1.099068,  1.198692,  -0.756014, -0.624271, -0.543815,
        -1.929007, 0.178418,  0.454656,  0.406123,  1.001768,  -0.185360,
        -1.428434, -1.360957, -0.927513, 0.094791,  1.394335,  1.005247};

    std::vector<float> mean_data = {
        0.43581513, 0.49090168
    };
    std::vector<float> var_data = {
        0.03025229, 0.11069085
    };
    std::vector<float> gamma_data = {
        1,1
    };
    std::vector<float> beta_data = {
        0, 0
    };

    auto input_tensor = graph->CreateTensor(input_spec);
    auto output_tensor = graph->CreateTensor(output_spec);
    auto mean = graph->CreateTensor(mean_spec, mean_data.data());
    auto var = graph->CreateTensor(var_spec, var_data.data());
    auto gamma = graph->CreateTensor(gamma_spec, gamma_data.data());
    auto beta = graph->CreateTensor(beta_spec, beta_data.data());

    float epsilon = 0.001;
    
    auto op = graph->CreateOperation<tim::vx::ops::BatchNorm>(epsilon);
    (*op).BindInputs({input_tensor, mean, var,gamma, beta}).BindOutputs({output_tensor});

    EXPECT_TRUE(graph->Compile());
    input_tensor->CopyDataToTensor(in_data.data(),
                                   in_data.size() * sizeof(float));
    EXPECT_TRUE(graph->Run());

    std::vector<float> output(golden.size());
    EXPECT_TRUE(output_tensor->CopyDataFromTensor(output.data()));

    for (uint32_t idx = 0; idx < golden.size(); idx++) {
      EXPECT_TRUE(std::abs(golden[idx] - output[idx]) < 0.01);
    }
}