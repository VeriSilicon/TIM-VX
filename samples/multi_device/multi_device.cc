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
#include "tim/vx/platform/native.h"
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
void print_topN(std::size_t size, std::shared_ptr<tim::vx::platform::ITensorHandle> handle) {
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
std::pair<std::shared_ptr<tim::vx::platform::IExecutable>, std::shared_ptr<tim::vx::platform::ITensorHandle>> generate_executable(
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
  auto devices = tim::vx::platform::NativeDevice::Enumerate();
  auto device0 = devices[0];
  std::shared_ptr<tim::vx::platform::IExecutor> executor0 = std::make_shared<tim::vx::platform::NativeExecutor> (device0);
  auto device1 = devices[1];
  std::shared_ptr<tim::vx::platform::IExecutor> executor1 = std::make_shared<tim::vx::platform::NativeExecutor> (device1);
  auto device2 = devices[2];
  std::shared_ptr<tim::vx::platform::IExecutor> executor2 = std::make_shared<tim::vx::platform::NativeExecutor> (device2);
  auto device3 = devices[3];
  std::shared_ptr<tim::vx::platform::IExecutor> executor3 = std::make_shared<tim::vx::platform::NativeExecutor> (device3);

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

  std::shared_ptr<tim::vx::platform::IExecutable> lenet_0, lenet_2, lenet_3, mobilenet_1, mobilenet_2, mobilenet_3, resnet50_0, resnet50_1;
  std::shared_ptr<tim::vx::platform::ITensorHandle> lenet_0_outhandle, lenet_2_outhandle, lenet_3_outhandle, mobilenet_1_outhandle, mobilenet_2_outhandle, mobilenet_3_outhandle,
    resnet50_0_outhandle, resnet50_1_outhandle;

  std::tie(lenet_0, lenet_0_outhandle) = generate_executable(executor0, lenet_construct_func, lenet_weight_file, lenet_input_files, lenet_input_bytes);
  std::tie(resnet50_0, resnet50_0_outhandle) = generate_executable(executor0, resnet50_construct_func, resnet50_weight_file, resnet50_input_files, resnet50_input_bytes);
  executor0->Submit(lenet_0, lenet_0);
  executor0->Submit(resnet50_0, lenet_0);

  std::tie(mobilenet_1, mobilenet_1_outhandle) = generate_executable(executor1, mobilenet_construct_func, mobilenet_weight_file, mobilenet_input_files, mobilenet_input_bytes);
  std::tie(resnet50_1, resnet50_1_outhandle) = generate_executable(executor1, resnet50_construct_func, resnet50_weight_file, resnet50_input_files, resnet50_input_bytes);
  auto executable_set1 = tim::vx::platform::CreateExecutableSet({mobilenet_1, resnet50_1});
  executor1->Submit(executable_set1, executable_set1);

  std::tie(lenet_2, lenet_2_outhandle) = generate_executable(executor2, lenet_construct_func, lenet_weight_file, lenet_input_files, lenet_input_bytes);
  std::tie(mobilenet_2, mobilenet_2_outhandle) = generate_executable(executor2, mobilenet_construct_func, mobilenet_weight_file, mobilenet_input_files, mobilenet_input_bytes);
  auto executable_set2 = tim::vx::platform::CreateExecutableSet({lenet_2, mobilenet_2});
  executor2->Submit(executable_set2, executable_set2);

  std::tie(lenet_3, lenet_3_outhandle) = generate_executable(executor3, lenet_construct_func, lenet_weight_file, lenet_input_files, lenet_input_bytes);
  std::tie(mobilenet_3, mobilenet_3_outhandle) = generate_executable(executor3, mobilenet_construct_func, mobilenet_weight_file, mobilenet_input_files, mobilenet_input_bytes);
  auto executable_set3 = tim::vx::platform::CreateExecutableSet({lenet_3, mobilenet_3});
  executor3->Submit(executable_set3, executable_set3);

  std::thread t0(executor_trigger, executor0);
  std::thread t1(executor_trigger, executor1);
  std::thread t2(executor_trigger, executor2);
  std::thread t3(executor_trigger, executor3);
  t0.join();
  t1.join();
  t2.join();
  t3.join();

  print_topN<float>(1 * 10, lenet_0_outhandle);
  print_topN<float>(1 * 10, lenet_2_outhandle);
  print_topN<float>(1 * 10, lenet_3_outhandle);
  print_topN<float>(1 * 1001, mobilenet_1_outhandle);
  print_topN<float>(1 * 1001, mobilenet_2_outhandle);
  print_topN<float>(1 * 1001, mobilenet_3_outhandle);
  print_topN<uint16_t>(1 * 1000, resnet50_0_outhandle);
  print_topN<uint16_t>(1 * 1000, resnet50_1_outhandle);
  return 0;
}
