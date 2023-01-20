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
#include <tuple>
#include <vector>
#include <assert.h>
#include <chrono>

#include "lenet.h"
#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/platform/platform.h"
#include "tim/vx/platform/native.h"

std::vector<uint8_t> input_data = {
    0,   0,   0,   0,   0,   0,   0,   0,   6,   0,   2,   0,   0,   8,   0,
    3,   0,   7,   0,   2,   0,   0,   0,   10,  0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   3,   1,   1,   0,   14,  0,   0,   3,   0,
    2,   4,   0,   0,   0,   3,   1,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   4,   3,   0,   0,   0,   5,   0,   4,   0,   0,
    0,   0,   10,  12,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   6,   5,   0,   2,   0,   9,   0,   12,  2,   0,   5,   1,   0,
    0,   2,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    3,   0,   33,  0,   0,   155, 186, 55,  17,  22,  0,   0,   3,   9,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    2,   0,   167, 253, 255, 235, 255, 240, 134, 36,  0,   6,   1,   4,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   6,   87,
    240, 251, 254, 254, 237, 255, 252, 191, 27,  0,   0,   5,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   19,  226, 255, 235,
    255, 255, 254, 242, 255, 255, 68,  12,  0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   4,   1,   58,  254, 255, 158, 0,   2,
    47,  173, 253, 247, 255, 65,  4,   1,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   162, 240, 248, 92,  8,   0,   13,  0,
    88,  249, 244, 148, 0,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   17,  64,  244, 255, 210, 0,   0,   1,   2,   0,   52,  223,
    255, 223, 0,   11,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   144, 245, 255, 142, 0,   4,   9,   0,   6,   0,   37,  222, 226,
    42,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   73,
    255, 243, 104, 0,   0,   0,   0,   11,  0,   0,   0,   235, 242, 101, 4,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   133, 245, 226,
    12,  4,   15,  0,   0,   0,   0,   24,  0,   235, 246, 41,  0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   236, 245, 152, 0,   10,
    0,   0,   0,   0,   6,   0,   28,  227, 239, 1,   6,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   227, 240, 53,  4,   0,   0,   24,
    0,   1,   0,   8,   181, 249, 177, 0,   2,   0,   0,   0,   0,   4,   0,
    6,   1,   5,   0,   0,   87,  246, 219, 14,  0,   0,   2,   0,   10,  7,
    0,   134, 255, 249, 104, 4,   0,   0,   0,   0,   0,   8,   0,   3,   0,
    0,   0,   4,   89,  255, 228, 0,   11,  0,   8,   14,  0,   0,   100, 250,
    248, 236, 0,   0,   8,   0,   0,   0,   0,   5,   0,   2,   0,   0,   2,
    6,   68,  250, 228, 6,   6,   0,   0,   1,   0,   140, 240, 253, 238, 51,
    31,  0,   3,   0,   0,   0,   0,   0,   0,   5,   0,   0,   2,   0,   26,
    215, 255, 119, 0,   21,  1,   40,  156, 233, 244, 239, 103, 0,   6,   6,
    0,   0,   0,   0,   0,   0,   0,   5,   0,   0,   0,   0,   0,   225, 251,
    240, 141, 118, 139, 222, 244, 255, 249, 112, 17,  0,   0,   8,   3,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   84,  245, 255, 247,
    255, 249, 255, 255, 249, 132, 11,  0,   9,   3,   1,   1,   0,   0,   0,
    0,   2,   0,   0,   1,   0,   0,   6,   1,   0,   166, 236, 255, 255, 248,
    249, 248, 72,  0,   0,   16,  0,   16,  0,   4,   0,   0,   0,   0,   0,
    0,   0,   6,   0,   0,   4,   0,   0,   20,  106, 126, 188, 190, 112, 28,
    0,   21,  0,   1,   2,   0,   0,   3,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,
};
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

int main(int argc, char** argv) {
  (void) argc, (void) argv;
  auto context0 = tim::vx::Context::Create();
  auto graph0 = lenet(context0);
  auto graph1 = lenet(context0);

  auto devices = tim::vx::platform::NativeDevice::Enumerate();
  auto device = devices[0];
  std::shared_ptr<tim::vx::platform::IExecutor> executor = std::make_shared<tim::vx::platform::NativeExecutor> (device);

  auto executable0 = tim::vx::platform::Compile(graph0, executor);  // compile to nbg
  auto input_handle0 = executable0->AllocateTensor(graph0->InputsTensor()[0]->GetSpec());
  auto output_handle0 = executable0->AllocateTensor(graph0->OutputsTensor()[0]->GetSpec());
  executable0->SetInput(input_handle0);
  executable0->SetOutput(output_handle0);
  input_handle0->CopyDataToTensor(input_data.data(), input_data.size());
  assert(executable0->Submit(executable0));
  executable0->Trigger();

  auto executable1 = tim::vx::platform::Compile(graph1, executor);  // compile to nbg
  auto input_handle1 = executable1->AllocateTensor(graph1->InputsTensor()[0]->GetSpec());
  auto output_handle1 = executable1->AllocateTensor(graph1->OutputsTensor()[0]->GetSpec());
  executable1->SetInput(input_handle1);
  executable1->SetOutput(output_handle1);
  input_handle1->CopyDataToTensor(input_data.data(), input_data.size());
  assert(executable1->Submit(executable0));
  executable1->Trigger();

  executor->Submit(executable0, executable0);
  executor->Submit(executable1, executable0);

  std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executables0;
  executables0.push_back(executable0);
  executables0.push_back(executable1);
  auto executable_set0 = tim::vx::platform::CreateExecutableSet(executables0);
  executor->Submit(executable_set0, executable_set0);
  executor->Trigger();

  std::vector<uint8_t> input_data0;
  input_data0.resize(28 * 28);
  if (!input_handle0->CopyDataFromTensor(input_data0.data())) {
    std::cout << "Copy intput data fail." << std::endl;
    return -1;
  }
  printTopN(input_data0.data(), input_data0.size(), 5);

  std::vector<float> output_data;
  output_data.resize(1 * 10);
  if (!output_handle0->CopyDataFromTensor(output_data.data())) {
    std::cout << "Copy output data fail." << std::endl;
    return -1;
  }
  printTopN(output_data.data(), output_data.size(), 5);

  std::vector<float> output_data1;
  output_data1.resize(1 * 10);
  if (!output_handle1->CopyDataFromTensor(output_data1.data())) {
    std::cout << "Copy output data fail." << std::endl;
    return -1;
  }
  printTopN(output_data1.data(), output_data1.size(), 5);

  return 0;
}
