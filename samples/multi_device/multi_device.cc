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
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstring>
#include <tuple>
#include <vector>
#include <assert.h>
#include <chrono>
#include <thread>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/platform/platform.h"
#include "vx_lenet.h"
#include "vx_mobilenet.h"
#include "vx_resnet50.h"

template <typename T>
static void printTopN(const T* prob, int outputCount, int topNum) {
  std::vector<std::tuple<int, T>> data;

  for (int i = 0; i < outputCount; i++) {
    data.push_back(std::make_tuple(i, prob[i]));
  }

  std::sort(data.begin(), data.end(),
    [](auto& a, auto& b) { return std::get<1>(a) > std::get<1>(b); });

  std::cout << " --- Top" << topNum << " ---" << std::endl;
  for (int i = 0; i < topNum; i++) {
    std::cout << std::setw(3) << std::get<0>(data[i]) << ": " << std::fixed
              << std::setprecision(6) << std::get<1>(data[i]) << std::endl;
  }
}

template <typename T>
void print_topN(std::size_t size, std::shared_ptr<tim::vx::platform::ITensorHandle> & handle) {
  std::vector<T> output_data;
  output_data.resize(size);
  if (!handle->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
  }
  printTopN(output_data.data(), output_data.size(), 5);
}

std::vector<std::vector<char>> load_input_data(std::vector<std::string> filenames, std::vector<uint32_t> input_size_bytes) {
  std::vector<std::vector<char>> Data;
  for (std::size_t i = 0; i < filenames.size(); i++) {
    std::ifstream fin(filenames[i], std::ios::in | std::ios::binary);
    if (fin) {
        std::vector<char> input_data;
        fin.seekg(0, std::ios::end);
        int size = fin.tellg();
        fin.seekg(0, std::ios::beg);
        char *buffer = new char[size];
        std::cout<<"File "<<filenames[i] <<" size:"<<size<<std::endl;
        fin.read(buffer, size);
        fin.close();
        input_data.assign(buffer, buffer + input_size_bytes[i]);
        Data.push_back(input_data);
        delete []buffer;
    }
   }
    return Data;
}

void executor_trigger(std::shared_ptr<tim::vx::platform::IExecutor> executor) {
  executor->Trigger();
}

auto context = tim::vx::Context::Create();
std::pair<std::shared_ptr<tim::vx::platform::IExecutable>, std::shared_ptr<tim::vx::platform::ITensorHandle>>
  generate_executable(
    std::shared_ptr<tim::vx::platform::IExecutor> executor,
    std::function<void(std::shared_ptr<tim::vx::Graph>, const char*)> construct_func,
    std::string weight_file,
    std::vector<std::string> input_files, tim::vx::ShapeType input_size_bytes) {
  auto graph = context->CreateGraph();
  const char* weight_file_c = weight_file.c_str();
  construct_func(graph, weight_file_c);
  auto input_data = load_input_data(input_files, input_size_bytes);
  auto executable = tim::vx::platform::Compile(graph, executor);  // compile to nbg
  auto input_handle = executable->AllocateTensor(graph->InputsTensor()[0]->GetSpec());
  auto output_handle = executable->AllocateTensor(graph->OutputsTensor()[0]->GetSpec());
  executable->SetInput(input_handle);
  executable->SetOutput(output_handle);
  input_handle->CopyDataToTensor(input_data[0].data(), input_data[0].size());
  return std::make_pair(executable, output_handle);
}

int main(int argc, char** argv) {
  (void) argc, (void) argv;
  auto devices = tim::vx::platform::IDevice::Enumerate();
  auto device0 = devices[0];
  auto total_core_count = device0->CoreCount();
  uint32_t core_index = 0;
  auto use_core_count = 1;
  std::vector<std::shared_ptr<tim::vx::platform::IExecutor>> executors;

  for(core_index = 0; core_index < total_core_count; core_index += use_core_count) {
    auto executor = device0->CreateExecutor(core_index,use_core_count, context);
    executors.push_back(executor);
  }

  auto root = std::getenv("TIM_VX_ROOT");
  assert(root != NULL);
  std::string ROOT(root);
  std::vector<std::string> lenet_input_files = {ROOT + "/samples/multi_device/lenet/lenet_input_1_1_28_28_uint8.bin"};
  auto lenet_input_bytes = acuitylite::lenet::input_bytes_list;
  auto lenet_weight_file = ROOT + "/samples/multi_device/lenet/lenet.export.data";
  std::function<void(std::shared_ptr<tim::vx::Graph>, const char*)> lenet_construct_func = acuitylite::lenet::construct_graph;

  std::vector<std::string> mobilenet_input_files = {ROOT + "/samples/multi_device/mobilenet/mobilenet_1_224_224_3_uint8.bin"};
  auto mobilenet_input_bytes = acuitylite::mobilenet::input_bytes_list;
  auto mobilenet_weight_file = ROOT + "/samples/multi_device/mobilenet/mobilenet.export.data";
  std::function<void(std::shared_ptr<tim::vx::Graph>, const char*)> mobilenet_construct_func = acuitylite::mobilenet::construct_graph;

  std::vector<std::string> resnet50_input_files = {ROOT + "/samples/multi_device/resnet50/resnet50_1_3_224_224_uint8.bin"};
  auto resnet50_input_bytes = acuitylite::resnet50::input_bytes_list;
  auto resnet50_weight_file = ROOT + "/samples/multi_device/resnet50/resnet50.export.data";
  std::function<void(std::shared_ptr<tim::vx::Graph>, const char*)> resnet50_construct_func = acuitylite::resnet50::construct_graph;

  auto excutor_cnt  = executors.size();

  //each excutor run 2 models.
  auto lenet = [&](std::shared_ptr<tim::vx::platform::IExecutor> executor) {
    return generate_executable(executor, lenet_construct_func, lenet_weight_file,
                               lenet_input_files, lenet_input_bytes);
  };
  auto resnet = [&](std::shared_ptr<tim::vx::platform::IExecutor> executor) {
     return generate_executable(executor, resnet50_construct_func, resnet50_weight_file,
                                resnet50_input_files, resnet50_input_bytes);
  };
  auto mobilenet = [&](std::shared_ptr<tim::vx::platform::IExecutor> executor) {
     return generate_executable(executor, mobilenet_construct_func, mobilenet_weight_file,
                                mobilenet_input_files, mobilenet_input_bytes);
  };
  std::vector<std::pair<std::shared_ptr<tim::vx::platform::IExecutable>,
              std::shared_ptr<tim::vx::platform::ITensorHandle>>> nets;
  for (size_t i = 0; i < excutor_cnt; i++) {
    if(i % 3 == 0) {
      //lenet + resnet
      nets.push_back(lenet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
      nets.push_back(resnet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
    }
    if(i % 3 == 1) {
      //resnet + mobilenet
      nets.push_back(resnet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
      nets.push_back(mobilenet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
    }
    if(i % 3 == 2) {
      //lenet + mobilenet
      nets.push_back(mobilenet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
      nets.push_back(lenet(executors[i]));
      executors[i]->Submit(nets.back().first, nets.back().first);
    }
  }
  std::vector<std::thread> threads;
  for(auto executor:executors) {
        threads.push_back(std::thread(executor_trigger, executor));
  }
  for(std::thread &t : threads) {
     t.join();
  }

for (auto net : nets) {
  auto size = net.second->GetSpec().GetElementNum();
  print_topN<float>(size, net.second);
}
  return 0;
}
