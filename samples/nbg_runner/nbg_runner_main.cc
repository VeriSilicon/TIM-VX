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
#include "tim/vx/ops/nbg.h"
#include "tim/vx/types.h"
#include "tim/utils/nbg_parser/nbg_parser.h"
#include "tim/utils/nbg_parser/gc_vip_nbg_format.h"
#include <fstream>
#include <iostream>
#include <assert.h>
#include <vector>
#include <map>

#ifdef __linux__
#include <time.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

using namespace std;

static uint64_t get_perf_count()
{
#if defined(__linux__) || defined(__ANDROID__) || defined(__QNX__) || defined(__CYGWIN__)
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t)((uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec * 1000000000);
#elif defined(_WIN32) || defined(UNDER_CE)
    LARGE_INTEGER ln;

    QueryPerformanceCounter(&ln);

    return (uint64_t)ln.QuadPart;
#endif
}

static map<_nbg_buffer_quantize_format_e, tim::vx::QuantType> QMap = {
    {NBG_BUFFER_QUANTIZE_NONE, tim::vx::QuantType::NONE},
    {NBG_BUFFER_QUANTIZE_AFFINE_ASYMMETRIC, tim::vx::QuantType::ASYMMETRIC},
};

static map<tim::vx::QuantType, string> QNameMap = {
    {tim::vx::QuantType::NONE, "NONE"},
    {tim::vx::QuantType::ASYMMETRIC, "ASYMMETRIC_PER_TENSOR"},
    {tim::vx::QuantType::SYMMETRIC_PER_CHANNEL, "SYMMETRIC_PER_CHANNEL"},
};

static map<_nbg_buffer_format_e, tim::vx::DataType> DMap = {
    {NBG_BUFFER_FORMAT_FP32, tim::vx::DataType::FLOAT32},
    {NBG_BUFFER_FORMAT_FP16, tim::vx::DataType::FLOAT16},
    {NBG_BUFFER_FORMAT_UINT8, tim::vx::DataType::UINT8},
    {NBG_BUFFER_FORMAT_INT8, tim::vx::DataType::INT8},
    {NBG_BUFFER_FORMAT_UINT16, tim::vx::DataType::UINT16},
    {NBG_BUFFER_FORMAT_INT16, tim::vx::DataType::INT16},
    {NBG_BUFFER_FORMAT_UINT32, tim::vx::DataType::UINT32},
    {NBG_BUFFER_FORMAT_INT32, tim::vx::DataType::INT32},
};

static map<tim::vx::DataType, string> DNameMap = {
    {tim::vx::DataType::FLOAT32, "FLOAT32"},
    {tim::vx::DataType::FLOAT16, "FLOAT16"},
    {tim::vx::DataType::UINT8, "UINT8"},
    {tim::vx::DataType::INT8, "INT8"},
    {tim::vx::DataType::UINT16, "UINT16"},
    {tim::vx::DataType::INT16, "INT16"},
    {tim::vx::DataType::UINT32, "UINT32"},
    {tim::vx::DataType::INT32, "INT32"},
};


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout<< "usage: ./nbg_runner network.nb" << std::endl;
        return -1;
    }

    ifstream nbg_file((const char*)argv[1]);
    assert(nbg_file);

    nbg_file.seekg(0, ios::end);
    int nbg_size = nbg_file.tellg();
    std::vector<char> nbg_buf(nbg_size);
    nbg_file.seekg(0, ios::beg);
    nbg_file.read( nbg_buf.data(), nbg_size );
    nbg_file.close();

    nbg_parser_data nbg = NBG_NULL;

    nbg_parser_init(nbg_buf.data(), nbg_size, &nbg);

    // Get Inputs
    int input_count = 0;
    nbg_parser_query_network(nbg, NBG_PARSER_NETWORK_INPUT_COUNT, &input_count,
                             sizeof(input_count));

    printf("Number of Inputs: %d\n", input_count);
    vector<tim::vx::TensorSpec> input_list;
    for (int i = 0; i < input_count; i++) {
        printf(" Input: %d\n", i);
        // Get type of input, only support tensor for now
        _nbg_buffer_type_e data_type;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_DATA_TYPE, &data_type,
                               sizeof(data_type));
        assert( data_type == NBG_BUFFER_TYPE_TENSOR );

        // Get Shape
        unsigned int dim_count;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION,
                               &dim_count, sizeof(dim_count));
        assert( dim_count <= MAX_NUM_DIMS );
        unsigned int dim_size[MAX_NUM_DIMS];
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_DIMENSIONS, dim_size,
                               sizeof(dim_size) * dim_count);
        tim::vx::ShapeType input_shape;
        printf("\tShape: { ");
        for (unsigned int j=0 ; j<dim_count; j++) {
            input_shape.push_back(dim_size[j]);
            printf("%u ", input_shape[j]);
        }
        printf("}\n");

        // Get Quantization Format
        _nbg_buffer_quantize_format_e quant_format;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_QUANT_FORMAT,
                               &quant_format, sizeof(quant_format));
        float tf_scale;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_SCALE, &tf_scale,
                               sizeof(tf_scale));
        int tf_zerop;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_ZERO_POINT, &tf_zerop,
                               sizeof(tf_zerop));  
        tim::vx::Quantization input_quant(QMap[quant_format], tf_scale, tf_zerop);

        // Get Data Format
        _nbg_buffer_format_e data_format;
        nbg_parser_query_input(nbg, i, NBG_PARSER_BUFFER_PROP_DATA_FORMAT, &data_format,
                               sizeof(data_format));
        printf("\tFormat: %s - ", DNameMap[DMap[data_format]].c_str());
        printf("%s(%f, %d)\n", QNameMap[QMap[quant_format]].c_str(), tf_scale, tf_zerop);
        tim::vx::TensorSpec input_tensor(DMap[data_format], input_shape, tim::vx::TensorAttribute::INPUT, input_quant);
        input_list.push_back(input_tensor);
        }

    // Get Outtputs
    int output_count = 0;
    nbg_parser_query_network(nbg, NBG_PARSER_NETWORK_OUTPUT_COUNT, &output_count,
                             sizeof(output_count));
    printf("Number of Outputs: %d\n", output_count);
    vector<tim::vx::TensorSpec> output_list;
    for (int i = 0; i < output_count; i++) {
        printf(" Output: %d\n", i);
        // Get type of output, only support tensor for now
        _nbg_buffer_type_e data_type;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_DATA_TYPE, &data_type,
                               sizeof(data_type));
        assert( data_type == NBG_BUFFER_TYPE_TENSOR );

        // Get Shape
        unsigned int dim_count;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_NUM_OF_DIMENSION,
                               &dim_count, sizeof(dim_count));
        assert( dim_count <= MAX_NUM_DIMS );
        unsigned int dim_size[MAX_NUM_DIMS];
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_DIMENSIONS, dim_size,
                               sizeof(dim_size) * dim_count);
        tim::vx::ShapeType output_shape;
        printf("\tShape: { ");
        for (unsigned int j=0 ; j<dim_count; j++) {
            output_shape.push_back(dim_size[j]);
            printf("%u ", output_shape[j]);
        }
        printf("}\n");

        // Get Quantization Format
        _nbg_buffer_quantize_format_e quant_format;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_QUANT_FORMAT,
                               &quant_format, sizeof(quant_format));
        float tf_scale;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_SCALE, &tf_scale,
                               sizeof(tf_scale));
        int tf_zerop;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_ZERO_POINT, &tf_zerop,
                               sizeof(tf_zerop));  
        tim::vx::Quantization output_quant(QMap[quant_format], tf_scale, tf_zerop);

        // Get Data Format
        _nbg_buffer_format_e data_format;
        nbg_parser_query_output(nbg, i, NBG_PARSER_BUFFER_PROP_DATA_FORMAT, &data_format,
                               sizeof(data_format));
        printf("\tFormat: %s - ", DNameMap[DMap[data_format]].c_str());
        printf("%s(%f, %d)\n", QNameMap[QMap[quant_format]].c_str(), tf_scale, tf_zerop);
        tim::vx::TensorSpec output_tensor(DMap[data_format], output_shape, tim::vx::TensorAttribute::OUTPUT, output_quant);
        output_list.push_back(output_tensor);
        }

    auto ctx = tim::vx::Context::Create();
    auto graph = ctx->CreateGraph();
    auto nbg_node = graph->CreateOperation<tim::vx::ops::NBG>(
        nbg_buf.data(), input_count, output_count);
    for(int i=0; i<input_count; i++) {
        auto input = graph->CreateTensor(input_list[i]);
        (*nbg_node).BindInput(input);
    }
    for(int i=0; i<output_count; i++) {
        auto output = graph->CreateTensor(output_list[i]);
        (*nbg_node).BindOutput(output);
    }

    uint64_t tmS, tmE;

    tmS = get_perf_count();
    assert(graph->Compile());
    tmE = get_perf_count();
    printf("Compile Time: %ldus\n", (tmE - tmS)/1000);

    tmS = get_perf_count();
    assert(graph->Run());
    tmE = get_perf_count();
    printf("Run Time: %ldus\n", (tmE - tmS)/1000);

}
