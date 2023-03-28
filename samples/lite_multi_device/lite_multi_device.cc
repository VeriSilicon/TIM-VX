
/****************************************************************************
*
*    Copyright (c) 2023 Vivante Corporation
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
#include "tim/vx/types.h"
#include "tim/vx/platform/native.h"
#include "tim/vx/platform/lite/lite_native.h"

int main() {
  //construct tim-vx graph
  auto ctx = tim::vx::Context::Create();
  auto graph = ctx->CreateGraph();

  tim::vx::ShapeType io_shape({2, 2});
  tim::vx::TensorSpec input_spec(tim::vx::DataType::INT32, io_shape,
                                 tim::vx::TensorAttribute::INPUT);
  tim::vx::TensorSpec output_spec(tim::vx::DataType::INT32, io_shape,
                                  tim::vx::TensorAttribute::OUTPUT);
  auto input_t0 = graph->CreateTensor(input_spec);
  auto input_t1 = graph->CreateTensor(input_spec);
  auto output_t = graph->CreateTensor(output_spec);

  auto add = graph->CreateOperation<tim::vx::ops::Add>();
  (*add).BindInputs({input_t0, input_t1}).BindOutputs({output_t});

  std::vector<int> data_vec_i0({1, 2, 3, 4});
  std::vector<int> data_vec_i1({4, 3, 2, 1});

  auto devices = tim::vx::platform::NativeDevice::Enumerate();
  auto device = devices[0];
  auto executor = std::make_shared<tim::vx::platform::LiteNativeExecutor>(device);
  auto executable = executor->Compile(graph);
  auto input0_handle = executable->AllocateTensor(input_spec);
  auto input1_handle = executable->AllocateTensor(input_spec);
  auto output_handle = executable->AllocateTensor(output_spec);
  executable->SetInput(input0_handle);
  executable->SetInput(input1_handle);
  executable->SetOutput(output_handle);
  input0_handle->CopyDataToTensor(data_vec_i0.data(),
                                  data_vec_i0.size() * sizeof(int));
  input1_handle->CopyDataToTensor(data_vec_i1.data(),
                                  data_vec_i1.size() * sizeof(int));
  executable->Submit(executable);
  executor->Trigger();

  int* data = (int*)malloc(4 * sizeof(int));

  output_handle->CopyDataFromTensor(data);

  //each output value should be "5" in this demo
  for (int i = 0; i < 4; ++i) {
    std::cout << "output value: " << data[i] << std::endl;
  }
  free(data);
  return 0;
}
