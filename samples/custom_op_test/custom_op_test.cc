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
#include "tim/vx/ops.h"
#include "custom_gemm.h"
#include <tuple>

void custom_gemm_single_test(){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType a_shape({6, 2});
    tim::vx::ShapeType b_shape({2, 6});
    tim::vx::ShapeType out_shape({2, 2});
    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                    a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::FLOAT32,
                    b_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> a_data = {
        1, 2, 3, 4, 5, 6,
        -1, -2, -3, -4, -5, -6
    };
    std::vector<float> b_data = {
        6, 5,
        4, 3,
        2, 1,
        -6, -5,
        -4, -3,
        -2, -1
    };
    std::vector<float> golden = {
        -36, -27,
        36, 27
    };

    a_tensor->CopyDataToTensor(a_data.data(), a_data.size() * sizeof(float));
    b_tensor->CopyDataToTensor(b_data.data(), b_data.size() * sizeof(float));

    tim::vx::ops::CustomGemm::ParamTuple tuple_list(2,6,6,0,0,1.0,0,1.0,0,1.0,0);

    auto op = graph->CreateOperation<tim::vx::ops::CustomGemm>(
        false,false,tuple_list);

    (*op).BindInputs({a_tensor, b_tensor}).BindOutputs({out_tensor});

    graph->Compile();
    graph->Run();

    std::vector<float> output(golden.size());
    out_tensor->CopyDataFromTensor(output.data());

    std::cout<<"the diff between golan and result:"<<std::endl;
    for(uint32_t i=0;i<output.size();i++){
        std::cout<<output[i] - golden[i]<<" ";
    }
    std::cout<<std::endl;
}

void custom_gemm_op_and_add_op_test(){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType a_shape({6, 2});
    tim::vx::ShapeType b_shape({6, 2});
    tim::vx::ShapeType c_shape({6, 2});
    tim::vx::ShapeType d_shape({2, 6});

    tim::vx::ShapeType out_shape({2, 2});

    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                    a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::FLOAT32,
                    b_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec c_spec(tim::vx::DataType::FLOAT32,
                    c_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec d_spec(tim::vx::DataType::FLOAT32,
                    d_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto c_tensor = graph->CreateTensor(c_spec);
    auto d_tensor = graph->CreateTensor(d_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> a_data = {
        0, 1, 2, 3, 4, 5,
        -1, -2, -3, -4, -5, -6
    };

    std::vector<float> b_data = {
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0
    };

    std::vector<float> d_data = {
        6, 5,
        4, 3,
        2, 1,
        -6, -5,
        -4, -3,
        -2, -1
    };
    std::vector<float> golden = {
        -36, -27,
        36, 27
    };

    a_tensor->CopyDataToTensor(a_data.data(), a_data.size() * sizeof(float));
    b_tensor->CopyDataToTensor(b_data.data(), b_data.size() * sizeof(float));
    d_tensor->CopyDataToTensor(d_data.data(), d_data.size() * sizeof(float));

    auto op_add = graph->CreateOperation<tim::vx::ops::AddN>(2);
    (*op_add).BindInputs({a_tensor, b_tensor}).BindOutputs({c_tensor});

    tim::vx::ops::CustomGemm::ParamTuple tuple_list(2,6,6,0,0,1.0,0,1.0,0,1.0,0);

    auto op_gemm = graph->CreateOperation<tim::vx::ops::CustomGemm>(
        false,false,tuple_list);

    (*op_gemm).BindInputs({c_tensor, d_tensor}).BindOutputs({out_tensor});

    graph->Compile();
    graph->Run();

    std::vector<float> output(golden.size());
    out_tensor->CopyDataFromTensor(output.data());
    std::cout<<"the diff between golan and result:"<<std::endl;
    for(uint32_t i=0;i<output.size();i++){
        std::cout<<output[i] - golden[i]<<" ";
    }
    std::cout<<std::endl;
}

void custom_gemm_op_and_custom_gemm_op_test(){
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    
    tim::vx::ShapeType a_shape({2, 2});
    tim::vx::ShapeType b_shape({2, 2});
    tim::vx::ShapeType c_shape({2, 2});
    tim::vx::ShapeType d_shape({2, 2});

    tim::vx::ShapeType out_shape({2, 2});

    tim::vx::TensorSpec a_spec(tim::vx::DataType::FLOAT32,
                    a_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec b_spec(tim::vx::DataType::FLOAT32,
                    b_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec c_spec(tim::vx::DataType::FLOAT32,
                    c_shape, tim::vx::TensorAttribute::TRANSIENT);
    tim::vx::TensorSpec d_spec(tim::vx::DataType::FLOAT32,
                    d_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT32,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);

    auto a_tensor = graph->CreateTensor(a_spec);
    auto b_tensor = graph->CreateTensor(b_spec);
    auto c_tensor = graph->CreateTensor(c_spec);
    auto d_tensor = graph->CreateTensor(d_spec);
    auto out_tensor = graph->CreateTensor(out_spec);

    std::vector<float> a_data = {
        1, 1, 1,1
    };

    std::vector<float> b_data = {
        1, 1, 1, 1
    };

    std::vector<float> d_data = {
        1,1,1,1
    };
    std::vector<float> golden = {
        4,4,4,4
    };

    a_tensor->CopyDataToTensor(a_data.data(), a_data.size() * sizeof(float));
    b_tensor->CopyDataToTensor(b_data.data(), b_data.size() * sizeof(float));
    d_tensor->CopyDataToTensor(d_data.data(), d_data.size() * sizeof(float));

    tim::vx::ops::CustomGemm::ParamTuple tuple_list(2,2,2,0,0,1.0,0,1.0,0,1.0,0);

    auto op_gemm = graph->CreateOperation<tim::vx::ops::CustomGemm>(
        false,false,tuple_list);

    (*op_gemm).BindInputs({a_tensor, b_tensor}).BindOutputs({c_tensor});

    auto op_gemm2 = graph->CreateOperation<tim::vx::ops::CustomGemm>(
        false,false,tuple_list);

    (*op_gemm2).BindInputs({c_tensor, d_tensor}).BindOutputs({out_tensor});

    graph->Compile();
    graph->Run();

    std::vector<float> output(golden.size());
    out_tensor->CopyDataFromTensor(output.data());
    std::cout<<"the diff between golan and result:"<<std::endl;
    for(uint32_t i=0;i<output.size();i++){
        std::cout<<output[i] - golden[i]<<" ";
    }
    std::cout<<std::endl; 
}

int main(){
    custom_gemm_single_test();
    custom_gemm_op_and_add_op_test();
    custom_gemm_op_and_custom_gemm_op_test();
    return 1;
}