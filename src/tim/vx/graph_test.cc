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

#include "gtest/gtest.h"

#include <vector>

TEST(graph, gen_binary_graph_with_empty_graph) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    size_t bin_size = -1;
    EXPECT_FALSE(graph->CompileToBinary(nullptr, &bin_size)) << "Can not generate binary graph if it is empty";
}

TEST(graph, gen_binary_graph_with_simple_add) {
    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();

    tim::vx::ShapeType io_shape({1,1,1,1});
    tim::vx::TensorSpec input_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::INPUT);
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT32, io_shape, tim::vx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec);
    auto input_t1 = graph->CreateTensor(input_spec);
    auto output_t = graph->CreateTensor(output_spec);

    auto add = graph->CreateOperation<tim::vx::ops::Add>();
    (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    float in = 1.0f;
    float expected_out = 2.0f;
    EXPECT_TRUE(input_t0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(input_t1->CopyDataToTensor(&in, sizeof(in)));

    EXPECT_TRUE(graph->Run());
    float output = 0.0f;
    EXPECT_TRUE(output_t->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);

    auto nbg_graph = ctx->CreateGraph();
    auto nbg_in0 = nbg_graph->CreateTensor(input_spec);
    auto nbg_in1 = nbg_graph->CreateTensor(input_spec);
    auto nbg_out = nbg_graph->CreateTensor(output_spec);

    EXPECT_TRUE(nbg_in0->CopyDataToTensor(&in, sizeof(in)));
    EXPECT_TRUE(nbg_in1->CopyDataToTensor(&in, sizeof(in)));

    auto nbg_node = nbg_graph->CreateOperation<tim::vx::ops::NBG>(
        (nbg_buf.data()), /*num_of_input*/ 2,
        /*num_of_output*/ 1);
    (*nbg_node).BindInputs({nbg_in0, nbg_in1}).BindOutputs({nbg_out});
    EXPECT_TRUE(nbg_graph->Compile());
    EXPECT_TRUE(nbg_graph->Run());

    output=0.0f;
    EXPECT_TRUE(nbg_out->CopyDataFromTensor(&output));
    EXPECT_EQ(output, expected_out);
}

// You can disable compile trace_test if only need replay
// #undef ENABLE_API_TRACE
#ifdef ENABLE_API_TRACE
#define API_TRACER_IMPLEMENTATION   // enable static members in api tracer
#define TARGET_NAMESPACE_NAME "tim::vx"
#include "tim/experimental/trace/trace_tvx.h"

namespace tvx = trace;

TEST(graph, trace_test) {
    // Replace all tim::vx name space with tvx, tvx can be alias of trace
    // namespace.
    auto ctx = tvx::Context::Create();
    auto graph = ctx->CreateGraph();

    tvx::ShapeType io_shape({1,2,2,1});
    tvx::TensorSpec input_spec(tvx::DataType::FLOAT32, io_shape, tvx::TensorAttribute::INPUT);
    tvx::TensorSpec output_spec(tvx::DataType::FLOAT32, io_shape, tvx::TensorAttribute::OUTPUT);
    auto input_t0 = graph->CreateTensor(input_spec);
    auto input_t1 = graph->CreateTensor(input_spec);
    auto input_t2 = graph->CreateTensor(input_spec);
    auto output_t0 = graph->CreateTensor(output_spec);

    auto reshape = graph->CreateOperation<tvx::ops::Reshape>(io_shape);
    (*reshape).BindInput(input_t0).BindOutput(input_t1);
    auto add = graph->CreateOperation<tvx::ops::Add>();
    (*add).BindInputs({input_t0, input_t2}).BindOutputs({output_t0});

    size_t bin_size = -1;
    EXPECT_TRUE(graph->CompileToBinary(nullptr, &bin_size));
    EXPECT_NE(bin_size, -1);
    std::vector<char> nbg_buf(bin_size);

    // generate binary graph does't require input data
    EXPECT_TRUE(graph->CompileToBinary(nbg_buf.data(), &bin_size));

    // binary graph compilation doesn't impact current graph's execution
    std::vector<float> in = {1.1f, 2.2f, 3.3f, 4.4f};
    std::vector<float> expected_out = {2.2f, 4.4f, 6.6f, 8.8f};;
    EXPECT_TRUE(input_t0->CopyDataToTensor(in.data(), sizeof(float) * in.size()));
    EXPECT_TRUE(input_t2->CopyDataToTensor(in.data(), sizeof(float) * in.size()));

    EXPECT_TRUE(graph->Run());
    std::vector<float> output(in.size());
    EXPECT_TRUE(output_t0->CopyDataFromTensor(output.data()));
    EXPECT_EQ(output, expected_out);

    // extra test for Quantization apis
    tvx::Quantization quant0;
    quant0.SetType(tvx::QuantType::ASYMMETRIC);
    quant0.SetChannelDim(1);
    quant0.SetScales(std::vector<float>({0.2, 0.3}));
    quant0.SetZeroPoints(std::vector<int32_t>({2, 3}));

}
#endif /* #ifdef ENABLE_API_TRACE */


/*******************************************************************************
 * How to replay a trace_log.cc:
 * 1. Copy trace_log.cc in the root dir of tim-vx, rename with trace_log.rpl.cc
 * 2. And copy the trace_bin.bin file to the runtime workspace,
 *    rename with trace_bin.rpl.bin
 * 3. (optional) Add compile and run api call for specific graph
 * 4. Set follows 0->1 and re-compile.
 ******************************************************************************/
#if 0
#define API_REPLAYER_IMPLEMENTATION // enable static members in api replayer
#include "tim/experimental/trace/replayer.h"
TEST(graph, replay_test) {
    #include "trace_log.rpl.cc"
    // Manual compile and run the selected graph if those api calls not exist.
    // Like:
    // graph_12->Compile();
    // graph_12->Run();
    // Last rebuild unit-test and execute this case with:
    // `$build/install/bin/unit-test --gtest_filter=*replay_test*`
}

#endif /* #if 0 */
