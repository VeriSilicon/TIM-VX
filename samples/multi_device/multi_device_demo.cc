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
#include <iostream>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/tensor.h"
#include "tim/vx/platform/native.h"

static void printTopN() {
}

int demo(int argc, char** argv) {
  (void) argc, (void) argv;
  std::vector<uint8_t> input_data = {};
  auto context = tim::vx::Context::Create();
  //         -->g1--g2-->
  //    g0-->|           |-->g5
  //         -->g3--g4-->
  std::shared_ptr<tim::vx::Graph> g0, g1, g2, g3, g4, g5;

  tim::vx::TensorSpec g0_input0, g0_output0, g1_output0, g2_output0, g3_output0, g4_output0, g5_output0;

  // query device and get executor of devcie
  auto devices = tim::vx::platform::NativeDevice::Enumerate();
  auto device = devices[0];
  std::shared_ptr<tim::vx::platform::IExecutor> executor = std::make_shared<tim::vx::platform::NativeExecutor> (device);

  // executable0
  auto executable0 = executor->Compile(g0);  // compile to nbg
  auto input_handle = executable0->AllocateTensor(g0_input0);
  executable0->SetInput(input_handle);  // set input_hanlde
  input_handle->CopyDataToTensor(input_data.data(), input_data.size());
  executable0->SetOutput(executable0->AllocateTensor(g0_output0)); // set output_handle
  // executable1
  auto executable1 = executor->Compile(g1);  // compile to nbg
  executable1->SetInput(executable1->AllocateTensor(g0_output0));  // set input_hanlde
  executable1->SetOutput(executable1->AllocateTensor(g1_output0)); // set output_handle
  // executable2
  auto executable2 = executor->Compile(g2);  // compile to nbg
  executable2->SetInput(executable2->AllocateTensor(g1_output0));  // set input_hanlde
  executable2->SetOutput(executable2->AllocateTensor(g2_output0)); // set output_handle
  // executable3
  auto executable3 = executor->Compile(g3);  // compile to nbg
  executable3->SetInput(executable3->AllocateTensor(g2_output0));  // set input_hanlde
  executable3->SetOutput(executable3->AllocateTensor(g3_output0)); // set output_handle
  // executable4
  auto executable4 = executor->Compile(g4);  // compile to nbg
  executable4->SetInput(executable4->AllocateTensor(g3_output0));  // set input_hanlde
  executable4->SetOutput(executable4->AllocateTensor(g4_output0)); // set output_handle
  // executable5
  auto executable5 = executor->Compile(g5);  // compile to nbg
  executable5->SetInput(executable5->AllocateTensor(g3_output0));  // set input_hanlde
  executable5->SetInput(executable5->AllocateTensor(g4_output0));  // set input_hanlde
  executable5->SetOutput(executable5->AllocateTensor(g5_output0)); // set output_handle

  /* 1. one way to run */
  executable0->Submit(executable0);
  executable1->Submit(executable0);  // executable1 run after executable0
  executable2->Submit(executable1);
  executable3->Submit(executable0);
  executable4->Submit(executable3);
  executable5->Submit(executable2);  // executable5 run after executable2
  executable5->Submit(executable4);  // executable5 run after executable4
  // trigger
  executor->Trigger();  // run all submitted executables

  /* 2. another way to run */
  // executable_set0
  std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executables0;
  executables0.push_back(executable0);
  auto executable_set0 = CreateExecutableSet(executables0);
  // executable_set1
  std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executables1;
  executables1.push_back(executable1);
  executables1.push_back(executable3);
  auto executable_set1 = CreateExecutableSet(executables1);
  // executable_set2
  std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executables2;
  executables2.push_back(executable2);
  executables2.push_back(executable4);
  auto executable_set2 = CreateExecutableSet(executables2);
  // executable_set3
  std::vector<std::shared_ptr<tim::vx::platform::IExecutable>> executables3;
  executables3.push_back(executable5);
  auto executable_set3 = CreateExecutableSet(executables3);
  // submit executaleSets
  executable_set0->Submit(executable_set0);
  executable_set1->Submit(executable_set0);
  executable_set2->Submit(executable_set1);
  executable_set3->Submit(executable_set2);
  // trigger
  executor->Trigger();  // run all submitted executableSets

  printTopN();

  return 0;
}
