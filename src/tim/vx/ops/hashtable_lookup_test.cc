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
#include "tim/vx/ops/hashtable_lookup.h"

#include "gtest/gtest.h"
TEST(HashtableLookup, int32) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    
    tim::vx::ShapeType in_shape({2});
    tim::vx::ShapeType key_shape({4});
    tim::vx::ShapeType value_shape({4});
    tim::vx::ShapeType out_shape({2});;
    tim::vx::ShapeType hit_shape({2});

    tim::vx::TensorSpec in_spec(tim::vx::DataType::INT32,
                            in_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec key_spec(tim::vx::DataType::INT32,
                            key_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec value_spec(tim::vx::DataType::INT32,
                            value_shape, tim::vx::TensorAttribute::INPUT);  
    tim::vx::TensorSpec out_spec(tim::vx::DataType::FLOAT16,
                            out_shape, tim::vx::TensorAttribute::OUTPUT);
    tim::vx::TensorSpec hit_spec(tim::vx::DataType::INT32,
                            hit_shape, tim::vx::TensorAttribute::OUTPUT);
    auto in_tensor = graph->CreateTensor(in_spec);
    auto key_tensor = graph->CreateTensor(key_spec);
    auto value_tensor = graph->CreateTensor(value_spec);
    auto out_tensor = graph->CreateTensor(out_spec);
    auto hit_tensor = graph->CreateTensor(hit_spec);

    std::vector<int32_t> in_data = {
       10, 66
    };
    std::vector<int32_t> key_data = {
       10,20,30,40
    };
    std::vector<int32_t> value_data = {
       7,8,9,10
    };
    std::vector<uint16_t> out_golden = {18176,0};//check for float16 output
    std::vector<int32_t> hit_golden = {1,0};//0 means not hit
    
    EXPECT_TRUE(in_tensor->CopyDataToTensor(in_data.data(), in_data.size() * sizeof(int32_t)));
    EXPECT_TRUE(key_tensor->CopyDataToTensor(key_data.data(), key_data.size() * sizeof(int32_t)));
    EXPECT_TRUE(value_tensor->CopyDataToTensor(value_data.data(), value_data.size() * sizeof(int32_t)));
    EXPECT_TRUE(out_tensor->CopyDataToTensor(out_golden.data(), out_golden.size()));
    EXPECT_TRUE(hit_tensor->CopyDataToTensor(hit_golden.data(), hit_golden.size() * sizeof(int32_t)));

    auto op = graph->CreateOperation<tim::vx::ops::HashtableLookup>();
    (*op).BindInputs({in_tensor, key_tensor, value_tensor}).BindOutputs({out_tensor, hit_tensor});
    
    EXPECT_TRUE(graph->Compile());
    EXPECT_TRUE(graph->Run());

    std::vector<uint16_t> output(out_golden.size());
    std::vector<int32_t> hit(hit_golden.size());
    EXPECT_TRUE(out_tensor->CopyDataFromTensor(output.data()));
    EXPECT_TRUE(hit_tensor->CopyDataFromTensor(hit.data()));
    EXPECT_EQ(out_golden, output);
    EXPECT_EQ(hit_golden, hit);
}