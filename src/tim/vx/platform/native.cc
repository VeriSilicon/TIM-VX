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
#include "tim/vx/platform/native.h"
#include "native_device_private.h"
#include "tim/vx/ops/nbg.h"

namespace tim {
namespace vx {
namespace platform {

std::shared_ptr<IExecutable> Compile(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<IExecutor>& executor) {
  return executor->Compile(graph);
}

std::shared_ptr<IExecutable> CreateExecutableSet(
    const std::vector<std::shared_ptr<IExecutable>>& executables) {
  ExecutableSet* executable_set = new ExecutableSet(executables);
  std::shared_ptr<IExecutable> executable(executable_set);
  return executable;
}

IDevice::device_id_t IDevice::Id() const { return device_id_; }

void IDevice::RemoteReset() {}

NativeDeviceImpl::NativeDeviceImpl(device_id_t id) {
  vip_device_ = std::make_unique<vip::IDevice>(id);
  device_id_ = id;
}

bool NativeDeviceImpl::Submit(const std::shared_ptr<Graph>& graph) {
  GraphImpl* graphimp =
      dynamic_cast<GraphImpl*>(graph.get());  // hack to downcast
  vsi_graph_v_.push_back(graphimp->graph());
  return true;
}

bool NativeDeviceImpl::Trigger(bool async, async_callback cb) {
  // extract graph from tasks
  (void)async;
  bool status = false;
  while (!vsi_graph_v_.empty()) {
    auto task = vsi_graph_v_.front();
    vsi_graph_v_.erase(vsi_graph_v_.begin());
    status = vip_device_->GraphSubmit(task, cb, NULL);
  }
  return status;
}

void NativeDeviceImpl::WaitDeviceIdle() { vip_device_->WaitThreadIdle(); }

bool NativeDeviceImpl::DeviceExit() { return vip_device_->ThreadExit(); }

std::vector<std::shared_ptr<IDevice>> NativeDevice::Enumerate() {
  std::vector<std::shared_ptr<IDevice>> device_v;
  device_id_t deviceCount = 0;
  vsi_nn_context_t context;
  context = vsi_nn_CreateContext();
  vxQueryContext(context->c, VX_CONTEXT_DEVICE_COUNT_VIV, &deviceCount,
                 sizeof(deviceCount));
  std::cout << "Device count = " << deviceCount << std::endl;
  for (device_id_t i = 0; i < deviceCount; i++) {
    IDevice* local_device = new NativeDeviceImpl(i);
    std::shared_ptr<IDevice> local_device_sp(local_device);
    device_v.push_back(local_device_sp);
  }
  vsi_nn_ReleaseContext(&context);
  return device_v;
}

std::shared_ptr<Graph> IExecutable::NBGraph() const { return nb_graph_; }

std::shared_ptr<IExecutor> IExecutable::Executor() const {
  auto executor = executor_.lock();
  if (!executor) {
    std::cout << "Executor unable to lock weak_ptr";
  }
  return executor;
}

NativeExecutable::NativeExecutable(const std::shared_ptr<IExecutor>& executor,
                                   const std::vector<char>& nb_buf,
                                   size_t inputs, size_t outputs) {
  CompileOption opt;
  opt.setDeviceId(executor->Device()->Id());

  executor_ = executor;
  context_ = executor->Contex();
  nb_graph_ = context_->CreateGraph(opt);

  nb_buf_ = nb_buf;
  nb_node_ = nb_graph_->CreateOperation<tim::vx::ops::NBG>(nb_buf_.data(),
                                                           inputs, outputs);
}

void NativeExecutable::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  nb_node_->BindInput(th->GetTensor());
}

void NativeExecutable::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  nb_node_->BindOutput(th->GetTensor());
}

void NativeExecutable::GetOutput(
    const std::vector<std::shared_ptr<ITensorHandle>>& th) {
  (void)th;
}

bool NativeExecutable::Submit(const std::shared_ptr<IExecutable>& ref,
                              bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = Executor()->Submit(executable, ref, after);
  return status;
}

bool NativeExecutable::Trigger(bool async) {
  (void)async;
  bool status = false;
  auto device = Executor()->Device();
  device->Submit(nb_graph_);
  status = device->Trigger();
  device->WaitDeviceIdle();
  return status;
}

std::shared_ptr<ITensorHandle> NativeExecutable::AllocateTensor(
    const TensorSpec& tensor_spec) {
  auto tensor = nb_graph_->CreateTensor(tensor_spec);
  ITensorHandle* tensor_handle = new NativeTensorHandle(tensor);
  std::shared_ptr<ITensorHandle> tensor_handle_sp(tensor_handle);
  return tensor_handle_sp;
}

bool NativeExecutable::Verify() { return nb_graph_->Compile(); }

ExecutableSet::ExecutableSet(
    const std::vector<std::shared_ptr<IExecutable>>& executables) {
  executables_ = executables;
  executor_ = executables[0]->Executor();
}

void ExecutableSet::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  (void)th;
}

void ExecutableSet::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  (void)th;
}

void ExecutableSet::GetOutput(
    const std::vector<std::shared_ptr<ITensorHandle>>& th) {
  (void)th;
}

bool ExecutableSet::Submit(const std::shared_ptr<IExecutable>& ref,
                           bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = Executor()->Submit(executable, ref, after);
  return status;
}

bool ExecutableSet::Trigger(bool async) {
  (void)async;
  bool status = false;
  auto device = Executor()->Device();
  for (auto executable : executables_) {
    device->Submit(executable->NBGraph());
  }
  status = device->Trigger();
  device->WaitDeviceIdle();
  return status;
}

std::shared_ptr<ITensorHandle> ExecutableSet::AllocateTensor(
    const TensorSpec& tensor_spec) {
  std::shared_ptr<ITensorHandle> tensor_handle_sp;
  (void)tensor_spec;
  return tensor_handle_sp;
}

std::vector<std::shared_ptr<IExecutable>> ExecutableSet::Executables() const {
  return executables_;
}

bool ExecutableSet::Verify() {
  bool status = false;
  for (auto executable : executables_) {
    status = executable->Verify();
  }
  return status;
}

std::shared_ptr<Context> IExecutor::Contex() const { return context_; }

NativeExecutor::NativeExecutor(const std::shared_ptr<IDevice>& device) {
  device_ = device;
  context_ = Context::Create();
}

NativeExecutor::NativeExecutor(const std::shared_ptr<IDevice>& device,
                               const std::shared_ptr<Context>& context) {
  device_ = device;
  context_ = context;
}

bool NativeExecutor::Submit(const std::shared_ptr<IExecutable>& executable,
                            const std::shared_ptr<IExecutable>& ref,
                            bool after) {
  bool success = false;
  success = executable->Verify();
  if (success == false) {
    std::cout << "Executable NBG compile failed";
    return false;
  }
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

bool NativeExecutor::Trigger(bool async) {
  (void)async;
  while (!tasks_.empty()) {
    auto task = tasks_.front();
    tasks_.erase(tasks_.begin());
    auto task_ = task.lock();
    if (!task_) {
      std::cout << "Task unable to lock weak_ptr";
    }
    task_->Trigger();
  }
  device_->WaitDeviceIdle();
  return true;
}

std::shared_ptr<IExecutable> NativeExecutor::Compile(
    const std::shared_ptr<Graph>& graph) {

  CompileOption option;
  option.setDeviceId(device_->Id());
  graph->SetCompileOption(option);

  size_t bin_size = -1;
  graph->CompileToBinary(nullptr, &bin_size);
  std::vector<char> nb_buf;
  nb_buf.resize(bin_size);
  size_t inputs = graph->InputsTensor().size();
  size_t outputs = graph->OutputsTensor().size();
  graph->CompileToBinary(nb_buf.data(), &bin_size);
  std::shared_ptr<IExecutor> this_sp = shared_from_this();
  IExecutable* executable =
      new NativeExecutable(this_sp, nb_buf, inputs, outputs);
  std::shared_ptr<IExecutable> executable_sp(executable);
  return executable_sp;
}

std::shared_ptr<IDevice> IExecutor::Device() const { return device_; }

std::shared_ptr<Tensor> ITensorHandle::GetTensor() const { return tensor_; }

NativeTensorHandle::NativeTensorHandle(const std::shared_ptr<Tensor>& tensor) {
  tensor_ = tensor;
}

bool NativeTensorHandle::CopyDataToTensor(const void* data,
                                          uint32_t size_in_bytes) {
  return tensor_->CopyDataToTensor(data, size_in_bytes);
}

bool NativeTensorHandle::CopyDataFromTensor(void* data) {
  return tensor_->CopyDataFromTensor(data);
}

}  // namespace platform
}  // namespace vx
}  // namespace tim