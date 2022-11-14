#include <algorithm>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/matmul.h"
#include "tim/vx/tensor.h"

#ifdef CHIP_FLOW
#include "mxtypes.h"
#include "mx.h"

#endif

typedef union {
	unsigned int u;
	float f;
} _fp32_t;

float fp16_to_fp32(const short in)
{
	const _fp32_t magic	 = {(254 - 15) << 23};
	const _fp32_t infnan = {(127 + 16) << 23};
	_fp32_t o;
	o.u = (in & 0x7fff) << 13;
	o.f *= magic.f;
	if (o.f >= infnan.f) {
		o.u |= 255 << 23;
	}
	o.u |= (in & 0x8000) << 16;
	return o.f;
}

int main(int argc, char** argv) {
    (void) argc, (void) argv;

    
    std::vector<uint8_t> input_data = {};   // 1*1792*3*3
    std::vector<uint8_t> matmul_b_data = {};   // 1795*512
    int input_num = 3*3*1792*1;
    input_data.resize(input_num);
    std::ifstream input_file("/home/xuke/ssd/vosp/source/tim-vx/samples/metax_343/input.bin", std::ios::in | std::ios::binary);
    for(int i=0; i<input_num; i++) {
        uint8_t tmp;
        input_file.read((char*)&tmp, sizeof(uint8_t));
        input_data[i] = tmp;
        // std::cout << (int)tmp << std::endl;
    }
    input_file.close();   

    int mat_b_num = 1792*512;
    matmul_b_data.resize(mat_b_num);
    input_file.open("/home/xuke/ssd/vosp/source/tim-vx/samples/metax_343/matmul_b.bin", std::ios::in | std::ios::binary);
    for(int i=0; i<mat_b_num; i++) {
        uint8_t tmp;
        input_file.read((char*)&tmp, sizeof(uint8_t));
        matmul_b_data[i] = tmp;
        // std::cout << (int)tmp << std::endl;
    }
    input_file.close();   

    auto context = tim::vx::Context::Create();
    auto graph = context->CreateGraph();

    tim::vx::ShapeType input_shape({3, 3, 1792, 1});
    tim::vx::Quantization input_quant(tim::vx::QuantType::ASYMMETRIC, 0.007585f,
                                    102);
    tim::vx::TensorSpec input_spec(tim::vx::DataType::UINT8, input_shape,
                                    tim::vx::TensorAttribute::INPUT, input_quant);
    auto input = graph->CreateTensor(input_spec);

    tim::vx::Quantization pool1_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.004590f, 129);
    tim::vx::TensorSpec pool1_output_spec(tim::vx::DataType::UINT8, {},
                                            tim::vx::TensorAttribute::TRANSIENT,
                                            pool1_output_quant);
    auto pool1_output = graph->CreateTensor(pool1_output_spec);

    tim::vx::Quantization reshape1_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.004590f, 129);
    tim::vx::TensorSpec reshape1_output_spec(tim::vx::DataType::UINT8, {},
                                            tim::vx::TensorAttribute::TRANSIENT,
                                            reshape1_output_quant);
    auto reshape1_output = graph->CreateTensor(reshape1_output_spec);

    tim::vx::Quantization reshape2_output_quant(tim::vx::QuantType::ASYMMETRIC,
                                           0.004590f, 129);
    tim::vx::TensorSpec reshape2_output_spec(tim::vx::DataType::UINT8, {},
                                            tim::vx::TensorAttribute::TRANSIENT,
                                            reshape2_output_quant);
    auto reshape2_output = graph->CreateTensor(reshape2_output_spec);

    tim::vx::ShapeType matmul_b_shape({512, 1792});
    tim::vx::Quantization matmul_b_quant(tim::vx::QuantType::ASYMMETRIC, 0.007585f,
                                    102);
    tim::vx::TensorSpec matmul_b_spec(tim::vx::DataType::UINT8, matmul_b_shape,
                                    tim::vx::TensorAttribute::INPUT, input_quant);
    auto matmul_b = graph->CreateTensor(matmul_b_spec);

    tim::vx::ShapeType output_shape({512, 1});
    tim::vx::TensorSpec output_spec(tim::vx::DataType::FLOAT16, output_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
    auto output = graph->CreateTensor(output_spec);

    auto pool1 = graph->CreateOperation<tim::vx::ops::Pool2d>(
        tim::vx::PoolType::AVG_ANDROID, tim::vx::PadType::NONE,
        std::array<uint32_t, 2>({3, 3}), std::array<uint32_t, 2>({1,1}));
    (*pool1).BindInputs({input}).BindOutputs({pool1_output});

    auto reshape1 = graph->CreateOperation<tim::vx::ops::Reshape>(
        std::vector<uint32_t>({1792,1,1,1}));
    (*reshape1).BindInputs({pool1_output}).BindOutputs({reshape1_output});

    auto reshape2 = graph->CreateOperation<tim::vx::ops::Reshape>(
        std::vector<uint32_t>({1792,1}));
    (*reshape2).BindInputs({reshape1_output}).BindOutputs({reshape2_output});

    auto matmul1 = graph->CreateOperation<tim::vx::ops::Matmul>();
    (*matmul1).BindInputs({reshape2_output, matmul_b}).BindOutputs({output});

#ifndef CHIP_FLOW

  if (!graph->Compile()) {
    std::cout << "Compile graph fail." << std::endl;
    return -1;
  }

  if (!input->CopyDataToTensor(input_data.data(), input_data.size())) {
    std::cout << "Copy input data fail." << std::endl;
    return -1;
  }

  if (!matmul_b->CopyDataToTensor(matmul_b_data.data(), matmul_b_data.size())) {
    std::cout << "Copy input data fail." << std::endl;
    return -1;
  }

  if (!graph->Run()) {
    std::cout << "Run graph fail." << std::endl;
    return -1;
  }

  std::vector<uint16_t> output_data_half;
  output_data_half.resize(1 * 512);
  if (!output->CopyDataFromTensor(output_data_half.data())) {
    std::cout << "Copy output data fail." << std::endl;
    return -1;
  }

  std::vector<float> output_data;
  output_data.resize(512);
  for (int i=0; i<512; i++) {
      output_data[i] = fp16_to_fp32(*((int16_t *)(&(output_data_half[i]))));
  }

  output_data_half.clear();
#endif

#ifdef CHIP_FLOW
  size_t bin_size = 0;
  bool succeed = graph->CompileToBinary(nullptr, &bin_size);
  if (!succeed || (bin_size <= 0)) {
    std::cout << "MACAVX compile engine failed!" << std::endl;
  }
  char* engine_buf = new char[bin_size];
  // Generate binary graph does't require input data
  succeed = graph->CompileToBinary(engine_buf, &bin_size);

  mcError_t status = mcSuccess;
  // create engine
  MXuint32 engineId;
  status = mcnnCreateEngine(&engineId);
  if (status != mcSuccess) {
    std::cout << "CreateEngine failed!" << std::endl;
  }
  // create mx context
  mcnnContext mx_context;
  status = mcnnCreateContext(engineId, &mx_context);
  if (status != mcSuccess) {
    std::cout << "CreateContext failed!" << std::endl;
  }

  // create mc stream
  mcStream_t stream;
  status = mcStreamCreate(&stream);
  if (status != mcSuccess) {
    std::cout << "Create mc Stream failed!" << std::endl;
  }
  
  // Add network into UMD driver before execution
  MXuint32 network_id = 0;
  MXNetworkTransInfo NetworkTransInfo;
  NetworkTransInfo.RunType = MXNN_RUN_LOW_LATENCY;
  NetworkTransInfo.TransType = MXNN_TRANS_BUFFER_TYPE;
  NetworkTransInfo.Buffer.Addr = (MXuint8*)(engine_buf);
  NetworkTransInfo.Buffer.Size = (MXuint32)(bin_size);  
  NetworkTransInfo.Buffer.Name = (MXuint8*)("AvgReshapMatmulSubgraph");
  status = mcnnDeployNetwork(engineId, &NetworkTransInfo, &network_id);
  if(status != mcSuccess){
    std::cout << "DeployNetwork failed!" << std::endl;
  }

  void* mx_input_bufs[2];
  void* mx_output_bufs[1];
  
  std::vector<uint16_t> output_data_half;
  output_data_half.resize(512);
  std::vector<float> output_data;
  output_data.resize(512);

  void* input_buf = input_data.data();
  void* matmul_b_buf = matmul_b_data.data();
  void* output_buf = output_data_half.data();
  
  int type_size = 1;
  auto size = 512*type_size;

  uint8_t* input_dla_addr;
  uint8_t* matmul_b_dla_addr;
  uint16_t* output_dla_addr;
  
  status = mcMalloc((void**)&input_dla_addr, size);
  status = mcMemcpyAsync((void*)input_dla_addr, input_buf, size, mcMemcpyHostToDevice, stream);
  status = mcStreamSynchronize(stream);

  status = mcMalloc((void**)&matmul_b_dla_addr, size);
  status = mcMemcpyAsync((void*)matmul_b_dla_addr, matmul_b_buf, size, mcMemcpyHostToDevice, stream);
  status = mcStreamSynchronize(stream);

  status = mcMalloc((void**)&output_dla_addr, size*2);
  mx_input_bufs[0] = (char*)input_dla_addr;
  mx_input_bufs[1] = (char*)matmul_b_dla_addr;
  mx_output_bufs[0] = (char*)output_dla_addr;
  status = mcnnExecute(mx_context, network_id, 1/* batch_size */, (void **)(mx_input_bufs), (void **)(mx_output_bufs), stream);
  if (status != mcSuccess) {
    std::cout << "ExecuteNetwork failed!" << std::endl;
  }
  
  status = mcMemcpyAsync((void*)output_buf, output_dla_addr, size*2, mcMemcpyDeviceToHost, stream);
  status = mcFree(output_dla_addr);
  status = mcFree(matmul_b_dla_addr);
  status = mcFree(input_dla_addr);

  for (int i=0; i<512; i++) {
      output_data[i] = fp16_to_fp32(*((int16_t *)(&(output_data_half[i]))));
  }
  output_data_half.clear();

#endif

for(int i=0; i<512; i++) {
    std::cout << "Output value: " << output_data[i] << std::endl;
}

}