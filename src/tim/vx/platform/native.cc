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
#include "native_private.h"
#include "context_private.h"
#include "tim/vx/ops/nbg.h"
#ifdef ENABLE_PLATFORM_LITE
#include "tim/vx/platform/lite/lite_native.h"
#endif

#include <cassert>
namespace tim {
namespace vx {
namespace platform {

std::shared_ptr<IExecutable> Compile(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<IExecutor>& executor) {
  return executor->Compile(graph);
}

NativeDeviceImpl::NativeDeviceImpl(device_id_t id, uint32_t core_count) {
  device_id_ = id;
  core_count_ = core_count;
}
std::vector<std::shared_ptr<IDevice>> IDevice::Enumerate() {
#ifdef ENABLE_PLATFORM_LITE
  auto devices = tim::vx::platform::LiteNativeDevice::Enumerate();
#else
  auto devices = tim::vx::platform::NativeDevice::Enumerate();
#endif
  return devices;
}

void IDevice::RemoteReset() {}

bool NativeDeviceImpl::Submit(const std::shared_ptr<Graph>& graph) {
  (void)graph;
  return true;
}

bool NativeDeviceImpl::Trigger(bool async, async_callback cb) {
  (void)async;
  (void)cb;
  return true;
}

void NativeDeviceImpl::WaitDeviceIdle() {}

bool NativeDeviceImpl::DeviceExit() { return true; }

std::shared_ptr<IExecutor> NativeDeviceImpl::CreateExecutor(const int32_t core_index,
                                                    const int32_t core_count,
                                                    const std::shared_ptr<Context>& context) {
  std::shared_ptr<IDevice> this_sp = shared_from_this();
  auto  executor = std::make_shared<NativeExecutorImpl>(this_sp, core_count,core_index,context);
  return executor;
}

std::vector<std::shared_ptr<IDevice>> NativeDevice::Enumerate() {
  std::vector<std::shared_ptr<IDevice>> device_v;
#ifdef VSI_DEVICE_SUPPORT
  vsi_nn_context_t context = vsi_nn_CreateContext();
  vsi_nn_device_t  vsi_devices[VSI_MAX_DEVICES] = {0};
  vsi_status status  = VSI_FAILURE;
  vsi_size_t deviceCount = 0;

  status  = vsi_nn_GetDevices(context,vsi_devices,&deviceCount);
  if(status != VSI_SUCCESS){
        VSILOGE("Get device count fail");
        return device_v;
  }

  for (vsi_size_t i = 0; i < deviceCount; i++) {
    vsi_size_t available_core_count = 0;
    vsi_nn_GetDeviceCoreCount(vsi_devices[i],&available_core_count);
    auto  local_device = std::make_shared<NativeDeviceImpl>(i,available_core_count);
    device_v.push_back(local_device);
  }
#else
#error "VSI device API is not supportted, please upgrade Vivant SDK version >= 6.4.22 && ovxlib >= 1.2.26 !");
#endif
  return device_v;
}

NativeExecutableImpl::NativeExecutableImpl(const std::shared_ptr<IExecutor>& executor,
                                   const std::vector<char>& nb_buf,
                                   size_t inputs, size_t outputs) {

  executor_ = executor;
  context_ = executor->Contex();
  nb_graph_ = context_->CreateGraph();

  nb_buf_ = nb_buf;
  nb_node_ = nb_graph_->CreateOperation<tim::vx::ops::NBG>(nb_buf_.data(),
                                                           inputs, outputs);
}

void NativeExecutableImpl::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  nb_node_->BindInput(th->GetTensor());
   input_handles_.push_back(th);
}

void NativeExecutableImpl::SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) {
    for (auto& t : ths) {
    SetInput(t);
  }
}

void NativeExecutableImpl::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  nb_node_->BindOutput(th->GetTensor());
  output_handles_.push_back(th);
}

void NativeExecutableImpl::SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) {
  for (auto& t : ths) {
    SetOutput(t);
  }

}

bool NativeExecutableImpl::Submit(const std::shared_ptr<IExecutable>& ref,
                              bool after) {
  bool status = false;
  std::shared_ptr<IExecutable> executable = shared_from_this();
  std::shared_ptr<NativeExecutorImpl> executor = std::dynamic_pointer_cast<NativeExecutorImpl>(executor_.lock());
  status = executor->Submit(executable, ref, after);
  return status;
}

bool NativeExecutableImpl::Trigger(bool async) {
  (void)async;
  bool status = nb_graph_->Run();
  return status;
}

std::shared_ptr<ITensorHandle> NativeExecutableImpl::AllocateTensor(const TensorSpec& tensor_spec,
                                                                    void* data, uint32_t size) {
  (void)size;
  auto tensor = nb_graph_->CreateTensor(tensor_spec,data);
  return std::make_shared<NativeTensorHandleImpl>(tensor);
}

bool NativeExecutableImpl::Verify() {
  std::shared_ptr<NativeExecutorImpl> executor = std::dynamic_pointer_cast<NativeExecutorImpl>(executor_.lock());
  bool success = executor->BindDevices(NBGraph());
  if (success == false) {
    VSILOGE("Executable bind device failed");
    return false;
  }
  success = nb_graph_->Compile();
  return success;
  }

NativeExecutorImpl::NativeExecutorImpl(const std::shared_ptr<IDevice>& device,
                               const int32_t core_count,
                               const int32_t core_index,
                               const std::shared_ptr<Context>& context) {
  device_ = device;
  if(!context) {
    context_ = Context::Create();
  } else {
    context_ = context;
  }
  auto fixed_core_count = core_count;
  int32_t fixed_core_index = core_index;
  int32_t total_core_count  =(int32_t)device_->CoreCount();
  if (fixed_core_index < 0) {
    fixed_core_index = 0;
  }
  if (fixed_core_index > total_core_count - 1) {
     VSILOGE("Core index is larger than total core count");
     assert(false);
  }
  if (fixed_core_count <= 0 ) {
    fixed_core_count = total_core_count - fixed_core_index;
  }

  if (fixed_core_index + fixed_core_count > total_core_count) {
    fixed_core_count = total_core_count - fixed_core_index;
    VSILOGW(
        "Core_index + core_count is larger than total core count. Fix core count to %d", fixed_core_count);
  }
  core_index_ = (uint32_t)fixed_core_index;
  core_count_ = (uint32_t)fixed_core_count;
#ifdef VSI_DEVICE_SUPPORT
  vsi_nn_device_t  vsi_devices[VSI_MAX_DEVICES] = {0};
  vsi_size_t num_devices = 0;
  auto ctx = dynamic_cast<ContextImpl*>(context_.get());
  vsi_nn_GetDevices(ctx->context(),vsi_devices,&num_devices);
  vsi_nn_CreateSubDevice(vsi_devices[device_->Id()],core_index_,core_count_,&sub_devices_);
#endif
}

bool NativeExecutorImpl::Submit(const std::shared_ptr<IExecutable>& executable,
                            const std::shared_ptr<IExecutable>& ref,
                            bool after) {
  bool success = false;
  success = executable->Verify();
  if(success == false) {
    VSILOGE("Executable NBG compile failed");
    return false;
  }
  if(executable == ref) {
    tasks_.push_back(executable);
    return true;
  }
  for(size_t i = 0; i < tasks_.size(); i++) {
    if(tasks_[i].lock() == ref) {
      if(after == true) {
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

bool NativeExecutorImpl::Trigger(bool async) {
  (void)async;
  bool ret = false;
  while(!tasks_.empty()) {
    auto task = tasks_.front();
    tasks_.erase(tasks_.begin());
    auto task_tmp = task.lock();
    if(!task_tmp) {
      VSILOGE("Task unable to lock weak_ptr");
       return false;
    }
    ret = task_tmp->Trigger();
  }
  device_->WaitDeviceIdle();
  return ret;
}

std::shared_ptr<IExecutable> NativeExecutorImpl::Compile(
    const std::shared_ptr<Graph>& graph) {
  bool ret = BindDevices(graph);
  if(!ret) {
    return nullptr;
  }
  size_t bin_size = -1;
  ret = graph->CompileToBinary(nullptr, &bin_size);
  if(!ret) {
    return nullptr;
  }
  std::vector<char> nb_buf;
  nb_buf.resize(bin_size);
  size_t inputs = graph->InputsTensor().size();
  size_t outputs = graph->OutputsTensor().size();
  ret = graph->CompileToBinary(nb_buf.data(), &bin_size);
  if(!ret) {
    return nullptr;
  }
  std::shared_ptr<NativeExecutorImpl> this_sp = shared_from_this();
  auto  executable = std::make_shared<NativeExecutableImpl>(this_sp, nb_buf,inputs,outputs);
  return executable;
}


bool NativeExecutorImpl::BindDevices(const std::shared_ptr<Graph>& graph){
  vsi_status status  = VSI_SUCCESS;
#ifdef VSI_DEVICE_SUPPORT
  GraphImpl* graphimp = dynamic_cast<GraphImpl*>(graph.get());
  status = vsi_nn_BindDevices(graphimp->graph(), 1, &sub_devices_);
#else
  (void)graph;
#endif
  if(status == VSI_SUCCESS) {
    return true;
  }
  else{
    return false;
  }
}


NativeTensorHandleImpl::NativeTensorHandleImpl(const std::shared_ptr<Tensor>& tensor) {
  tensor_ = tensor;
  spec_ = tensor->GetSpec();
}

bool NativeTensorHandleImpl::CopyDataToTensor(const void* data,
                                          uint32_t size_in_bytes) {
  return tensor_->CopyDataToTensor(data, size_in_bytes);
}

bool NativeTensorHandleImpl::CopyDataFromTensor(void* data) {
  return tensor_->CopyDataFromTensor(data);
}

}  // namespace platform
}  // namespace vx
}  // namespace tim
