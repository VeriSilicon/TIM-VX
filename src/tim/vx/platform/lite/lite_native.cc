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
#include "tim/vx/platform/lite/lite_native.h"

#include <cassert>

#include "tim/vx/graph.h"
#include "graph_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace platform {
LiteNativeExecutor::LiteNativeExecutor(const std::shared_ptr<IDevice>& device) {
  device_ = device;
  context_ = Context::Create();
  database_ = VIP_NULL;

  vip_init();
  vip_query_database(&database_);
  nbg_linker_init(database_);
}

LiteNativeExecutor::~LiteNativeExecutor() {
  nbg_destroy_task(task_descriptor_);
  nbg_linker_destroy();
  vip_destroy();
}

bool LiteNativeExecutor::Submit(const std::shared_ptr<IExecutable>& executable,
                                const std::shared_ptr<IExecutable>& ref,
                                bool after) {
  bool success = false;
  if (executable == ref) {
    tasks_.push_back(executable);
    return true;
  }
  for (size_t i = 0; i < tasks_.size(); i++) {
    if (tasks_[i].lock() == ref) {
      if (after == true) {
        tasks_.insert(tasks_.begin() + i + 1, executable);
        success = true;
        break;
      } else {
        tasks_.insert(tasks_.begin() + i, executable);
        success = true;
        break;
      }
    }
  }
  return success;
}

bool LiteNativeExecutor::Trigger(bool async) {
  (void)async;
  vip_status_e status = VIP_SUCCESS;
  std::vector<vip_network> networks;
  for (auto exe : tasks_) {
    auto task = exe.lock();
    task->Verify();
    vip_network& network =
        std::dynamic_pointer_cast<LiteNativeExecutable>(task)->network_;
    networks.push_back(std::move(network));
  }
  status = nbg_create_task(networks.size(), networks.data(), &task_descriptor_);
  if (status != VIP_SUCCESS) {
    VSILOGE("create task descriptor fail");
    return false;
  }
  status = vip_trigger_task(task_descriptor_);
  if (status != VIP_SUCCESS) {
    VSILOGE("trigger task descriptor fail");
    return false;
  }
  status = vip_wait_task(task_descriptor_);
  if (status != VIP_SUCCESS) {
    VSILOGE("wait task descriptor fail");
    // nbg_gen_capture(networks.size(), networks.data());
    return false;
  }
  return true;
}

std::shared_ptr<IExecutable> LiteNativeExecutor::Compile(
    const std::shared_ptr<Graph>& graph) {
  GraphImpl* graphimp = dynamic_cast<GraphImpl*>(graph.get());
  IDevice::device_id_t id = device_->Id();
  vxSetGraphAttribute(graphimp->graph()->g, VX_GRAPH_DEVICE_INDEX_VIV,
                      (void*)(&id), sizeof(id));
  size_t bin_size = -1;
  graph->CompileToBinary(nullptr, &bin_size);
  std::vector<char> nb_buf;
  nb_buf.resize(bin_size);
  graph->CompileToBinary(nb_buf.data(), &bin_size);
  return std::make_shared<LiteNativeExecutable>(shared_from_this(), nb_buf);
}

LiteNativeExecutable::LiteNativeExecutable(
    const std::shared_ptr<IExecutor>& executor,
    const std::vector<char>& nb_buf) {
  executor_ = executor;
  context_ = executor->Contex();
  nb_graph_ = context_->CreateGraph();
  nbg_create_network(nb_buf.data(), nb_buf.size(),
                     VIP_CREATE_NETWORK_FROM_MEMORY, &network_);
  input_count_ = 0;
  output_count_ = 0;
  coeff_ = nullptr;
  command_ = nullptr;
  memory_pool_ = nullptr;
  others_ = nullptr;
  pre_command_ = nullptr;

  /* prepare vip network */
  vip_status_e status = VIP_SUCCESS;
  nbg_network_memory_size_t buffer_size;
  nbg_network_memory_buffer_t buffer;
  vip_memory_t coeff_buffer;
  vip_memory_t cmd_buffer;
  vip_memory_t pre_cmd_buffer;
  vip_memory_t pool_buffer;
  vip_memory_t others_buffer;
  nbg_query_network(network_, VIP_NETWORK_PROP_MEMORY_SIZE, &buffer_size);

  vip_allocate_videomemory(buffer_size.coeff, &coeff_);
  vip_allocate_videomemory(buffer_size.command, &command_);
  vip_allocate_videomemory(buffer_size.memory_pool, &memory_pool_);
  vip_allocate_videomemory(buffer_size.others, &others_);
  vip_allocate_videomemory(buffer_size.pre_command, &pre_command_);

  SetBuffer(&coeff_buffer, coeff_);
  SetBuffer(&cmd_buffer, command_);
  SetBuffer(&pre_cmd_buffer, pre_command_);
  SetBuffer(&pool_buffer, memory_pool_);
  SetBuffer(&others_buffer, others_);

  buffer.coeff = &coeff_buffer;
  buffer.command = &cmd_buffer;
  buffer.memory_pool = &pool_buffer;
  buffer.others = &others_buffer;
  buffer.pre_command = &pre_cmd_buffer;
  buffer.dma_command = nullptr;
  status = nbg_prepare_network(network_, &buffer);

  vip_flush_videomemory(coeff_, VIP_BUFFER_OPER_TYPE_FLUSH);
  vip_flush_videomemory(command_, VIP_BUFFER_OPER_TYPE_FLUSH);
  vip_flush_videomemory(pre_command_, VIP_BUFFER_OPER_TYPE_FLUSH);
  vip_flush_videomemory(memory_pool_, VIP_BUFFER_OPER_TYPE_FLUSH);
  vip_flush_videomemory(others_, VIP_BUFFER_OPER_TYPE_FLUSH);

  if (status != VIP_SUCCESS) {
    VSILOGE("failed to prepare network");
    assert(false);
  }
}

LiteNativeExecutable::~LiteNativeExecutable() {
  nbg_finish_network(network_);
  nbg_destroy_network(network_);
  if (coeff_) {
    vip_free_videomemory(coeff_);
    coeff_ = nullptr;
  }
  if (command_) {
    vip_free_videomemory(command_);
    command_ = nullptr;
  }
  if (memory_pool_) {
    vip_free_videomemory(memory_pool_);
    memory_pool_ = nullptr;
  }
  if (others_) {
    vip_free_videomemory(others_);
    others_ = nullptr;
  }
  if (pre_command_) {
    vip_free_videomemory(pre_command_);
    pre_command_ = nullptr;
  }
}

void LiteNativeExecutable::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  vip_status_e status = VIP_SUCCESS;
  gcvip_videomemory_t* mem =
      std::dynamic_pointer_cast<LiteNativeTensorHandle>(th)->tensor_buffer_;
  vip_memory_t buffer;
  SetBuffer(&buffer, mem);

  status = nbg_set_input(network_, input_count_, &buffer);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to set input: %d", input_count_);
    assert(false);
  }
  ++input_count_;
}

void LiteNativeExecutable::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  vip_status_e status = VIP_SUCCESS;
  gcvip_videomemory_t* mem =
      std::dynamic_pointer_cast<LiteNativeTensorHandle>(th)->tensor_buffer_;
  vip_memory_t buffer;
  SetBuffer(&buffer, mem);

  status = nbg_set_output(network_, output_count_, &buffer);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to set output: %d", output_count_);
    assert(false);
  }
  ++output_count_;
}

void LiteNativeExecutable::GetOutput(
    const std::vector<std::shared_ptr<ITensorHandle>>& th) {
  (void)th;
}

bool LiteNativeExecutable::Submit(const std::shared_ptr<IExecutable>& ref,
                                  bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = Executor()->Submit(executable, ref, after);
  return status;
}

bool LiteNativeExecutable::Trigger(bool async) {
  (void)async;
  return false;
}

bool LiteNativeExecutable::Verify() {
  int32_t input_count = 0;
  nbg_query_network(network_, VIP_NETWORK_PROP_INPUT_COUNT, &input_count);
  if (input_count != input_count_) {
    VSILOGE("input count mismatch, required: %d, provided: %d", input_count,
            input_count_);
    return false;
  }
  int32_t output_count = 0;
  nbg_query_network(network_, VIP_NETWORK_PROP_OUTPUT_COUNT, &output_count);
  if (output_count != output_count_) {
    VSILOGE("output count mismatch, required: %d, provided: %d", output_count,
            output_count_);
    return false;
  }

  return true;
}

std::shared_ptr<ITensorHandle> LiteNativeExecutable::AllocateTensor(
    const TensorSpec& tensor_spec) {
  auto tensor = nb_graph_->CreateTensor(tensor_spec);
  return std::make_shared<LiteNativeTensorHandle>(tensor);
}

void LiteNativeExecutable::SetBuffer(vip_memory_t* dst,
                                     gcvip_videomemory_t* src) {
  if (dst && src) {
    dst->cpu_logical = src->cpu_logical;
    dst->npu_physical = src->npu_physical;
    dst->size = src->size;
  }
}

LiteNativeTensorHandle::LiteNativeTensorHandle(
    const std::shared_ptr<Tensor>& tensor) {
  tensor_ = tensor;
  uint32_t size = tensor->GetSpec().GetByteSize();
  vip_allocate_videomemory(size, &tensor_buffer_);
}

LiteNativeTensorHandle::~LiteNativeTensorHandle() {
  if (tensor_buffer_) {
    vip_free_videomemory(tensor_buffer_);
    tensor_buffer_ = nullptr;
  }
}

bool LiteNativeTensorHandle::CopyDataToTensor(const void* data,
                                              uint32_t size_in_bytes) {
  memcpy(tensor_buffer_->cpu_logical, data, size_in_bytes);
  return true;
}

bool LiteNativeTensorHandle::CopyDataFromTensor(void* data) {
  memcpy(data, tensor_buffer_->cpu_logical, tensor_buffer_->size);
  return true;
}

}  // namespace platform
}  // namespace vx
}  // namespace tim
