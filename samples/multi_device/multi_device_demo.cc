#include <algorithm>
#include <iostream>
#include <vector>

#include "tim/vx/context.h"
#include "tim/vx/graph.h"
#include "tim/vx/operation.h"
#include "tim/vx/tensor.h"
#include "tim/vx/LocalTensorHandle.h"
#include "tim/vx/LocalDevice.h"
#include "tim/vx/LocalExecutor.h"
#include "tim/vx/LocalExecutable.h"
#include "tim/vx/IExecutableSet.h"

static void printTopN() {
}

int main(int argc, char** argv) {
  (void) argc, (void) argv;
  std::vector<uint8_t> input_data = {};
  auto context = tim::vx::Context::Create();
  //         -->g1--g2-->
  //    g0-->|           |-->g5
  //         -->g3--g4-->
  std::shared_ptr<tim::vx::Graph> g0, g1, g2, g3, g4, g5;

  tim::vx::TensorSpec g0_input0, g0_output0, g1_output0, g2_output0, g3_output0, g4_output0, g5_output0;

  // query device and get executor of devcie
  auto devices = tim::vx::LocalDevice::Enumerate();
  auto device = devices[0];
  std::shared_ptr<tim::vx::LocalExecutor> executor= std::make_shared<tim::vx::LocalExecutor> (device);

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
  std::vector<std::shared_ptr<tim::vx::IExecutable>> executables0;
  executables0.push_back(executable0);
  auto executable_set0 = CreateIExecutableSet(executables0);
  // executable_set1
  std::vector<std::shared_ptr<tim::vx::IExecutable>> executables1;
  executables1.push_back(executable1);
  executables1.push_back(executable3);
  auto executable_set1 = CreateIExecutableSet(executables1);
  // executable_set2
  std::vector<std::shared_ptr<tim::vx::IExecutable>> executables2;
  executables2.push_back(executable2);
  executables2.push_back(executable4);
  auto executable_set2 = CreateIExecutableSet(executables2);
  // executable_set3
  std::vector<std::shared_ptr<tim::vx::IExecutable>> executables3;
  executables3.push_back(executable5);
  auto executable_set3 = CreateIExecutableSet(executables3);
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
