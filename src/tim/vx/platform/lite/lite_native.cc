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
#include "lite_native_private.h"

#include <cassert>
#include "tim/vx/graph.h"
#include "graph_private.h"
#include "context_private.h"

namespace tim {
namespace vx {
namespace platform {

  LiteNetwork::LiteNetwork(vip_create_network_param_t& param) {
    vip_create_network(&param, sizeof(param), &network_);
  }
  vip_status_e LiteNetwork::Query(vip_enum property, void* value) {
    return vip_query_network(network_, property, value);
  }
  vip_status_e LiteNetwork::Set(vip_enum property, void* value) {
     return vip_set_network(network_, property, value);
  }
  vip_status_e LiteNetwork::Prepare() {
      return vip_prepare_network(network_);
  }
   vip_status_e LiteNetwork::Run() {return vip_run_network(network_);}

   vip_status_e LiteNetwork::Trigger() {return vip_trigger_network(network_);}

   vip_status_e LiteNetwork::Wait() {return vip_wait_network(network_);}

   vip_status_e LiteNetwork::Cancel() {return vip_cancel_network(network_);}

   vip_status_e LiteNetwork::QueryInput(vip_uint32_t index, vip_enum property, void* value) {
    return vip_query_input(network_, index, property,value);
  }

  vip_status_e LiteNetwork::QueryOutput(vip_uint32_t index, vip_enum property, void* value) {
    return vip_query_output(network_, index, property, value);
  }

  vip_status_e LiteNetwork::SetInput(vip_uint32_t index, std::shared_ptr<ITensorHandle> input) {
      vip_buffer buffer =
        std::dynamic_pointer_cast<LiteNativeTensorHandleImpl>(input)->GetBuffer();
    return vip_set_input(network_, index, buffer);
  }

  vip_status_e LiteNetwork::SetOutput(vip_uint32_t index, std::shared_ptr<ITensorHandle> output) {
      vip_buffer buffer =
        std::dynamic_pointer_cast<LiteNativeTensorHandleImpl>(output)->GetBuffer();
     return vip_set_output(network_, index, buffer);
  }

  LiteNetwork::~LiteNetwork(){
    vip_finish_network(network_);
    vip_destroy_network(network_);
  }

bool LiteNativeDevice::vip_initialized = false;

LiteNativeDeviceImpl::LiteNativeDeviceImpl(device_id_t id,uint32_t core_count) {
  device_id_ = id;
  core_count_ = core_count;
 }

bool LiteNativeDeviceImpl::Submit(const std::shared_ptr<Graph>& graph) {
  (void)graph;
  return true;
}

bool LiteNativeDeviceImpl::Trigger(bool async, async_callback cb) {
  (void)async;
  (void)cb;
  return true;
}
void LiteNativeDeviceImpl::WaitDeviceIdle() {}

bool LiteNativeDeviceImpl::DeviceExit() {return false;}

std::shared_ptr<IExecutor> LiteNativeDeviceImpl::CreateExecutor(const int32_t core_index,
                                                    const int32_t core_count,
                                                    const std::shared_ptr<Context>& context) {
  std::shared_ptr<IDevice> this_sp = shared_from_this();
  auto executor = std::make_shared<LiteNativeExecutorImpl>(this_sp, core_count, core_index, context);
  return executor;
}

std::vector<std::shared_ptr<IDevice>> LiteNativeDevice::Enumerate() {
  std::vector<std::shared_ptr<IDevice>> device_v;
  device_id_t deviceCount = 0;
  std::vector<uint32_t> core_count;
  uint32_t version = 0;
  if( !LiteNativeDevice::vip_initialized ) {
    vip_status_e status = vip_init();
    if(status != VIP_SUCCESS) {
      VSILOGE("Initialize viplite driver fail");
      return device_v;
    }
    LiteNativeDevice::vip_initialized = true;
  }
  version = vip_get_version();
  if (version >= 0x00010601 ) {
      vip_query_hardware(VIP_QUERY_HW_PROP_DEVICE_COUNT, sizeof(uint32_t), &deviceCount);
      core_count.resize(deviceCount);
      vip_query_hardware(VIP_QUERY_HW_PROP_CORE_COUNT_EACH_DEVICE,
      sizeof(uint32_t) * core_count.size(), core_count.data());
  }

  for (device_id_t i = 0; i < deviceCount; i++) {
    auto  local_device = std::make_shared<LiteNativeDeviceImpl>(i, core_count.at(i));
    device_v.push_back(local_device);
  }
  return device_v;
}

int LiteNativeExecutorImpl::executor_count = 0;

LiteNativeExecutorImpl::LiteNativeExecutorImpl(const std::shared_ptr<IDevice>& device,
  const int32_t core_count, const int32_t core_index, const std::shared_ptr<Context>& context)
 {
  device_ = device;
  context_ = context;
  if(context_ == nullptr) {
    context_ = tim::vx::Context::Create();
  }
  auto fixed_core_count = core_count;
  int32_t fixed_core_index = core_index;
  vip_status_e status  = VIP_SUCCESS;
  if( !LiteNativeDevice::vip_initialized ) {
     status = vip_init();
     if(status != VIP_SUCCESS){
      throw "Initialize viplite driver fail";
      }
  }
  int32_t total_core_count  = (int32_t)device->CoreCount();
  if (fixed_core_index < 0)
  {
    fixed_core_index = 0;
  }
  if (fixed_core_index > total_core_count - 1){
     throw "Core index is larger than total core count.";
  }
  if (fixed_core_count <= 0 ) {
    fixed_core_count = total_core_count - fixed_core_index;
  }

  if (fixed_core_index + fixed_core_count > total_core_count) {
    fixed_core_count = total_core_count - fixed_core_index;
    VSILOGW(
        "Core_index + core_count is larger than total core count. Fix core "
        "count to %d",
        fixed_core_count);
  }
  core_index_ = (uint32_t)fixed_core_index;
  core_count_ = (uint32_t)fixed_core_count;

#ifdef VSI_DEVICE_SUPPORT
  vsi_nn_device_t  vsi_devices[VSI_MAX_DEVICES] = {0};
  vsi_size_t num_devices = 0;
  vsi_size_t available_core_count = 0;
  auto ctx = dynamic_cast<ContextImpl*>(context_.get());
  vsi_nn_GetDevices(ctx->context(), vsi_devices, &num_devices);

  //Always use device 0 to compile NBG.
  vsi_nn_GetDeviceCoreCount(vsi_devices[0], &available_core_count);

  if(core_index_ + core_count_ > (uint32_t)available_core_count) {
      VSILOGE("the used core count is larger than compiler available core count");
      assert(false);
  }
  vsi_nn_CreateSubDevice(vsi_devices[0], core_index_, core_count_, &sub_device_);
#else
  VSILOGE("device is not supported!");
  assert(false);
#endif

  executor_count++;
}

LiteNativeExecutorImpl::~LiteNativeExecutorImpl() {
#ifdef VSI_DEVICE_SUPPORT
  if(sub_device_)
     vsi_nn_ReleaseDevice(&sub_device_);
#endif
  executor_count--;
  if(executor_count <1)
    vip_destroy();
}

bool LiteNativeExecutorImpl::Submit(const std::shared_ptr<IExecutable>& executable,
                                const std::shared_ptr<IExecutable>& ref,
                                bool after) {
  bool success = false;
  success = executable->Verify();
  if (success == false) {
    VSILOGE("Executable NBG compile failed");
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

bool LiteNativeExecutorImpl::Trigger(bool async) {
  (void)async;
  while (!tasks_.empty()) {
    auto task = tasks_.front();
    tasks_.erase(tasks_.begin());
    auto task_tmp = task.lock();
    if (!task_tmp) {
      VSILOGE("Task is empty");
      return false;
    }
    task_tmp->Trigger();
  }
  return true;
}

std::shared_ptr<IExecutable> LiteNativeExecutorImpl::Compile(
    const std::shared_ptr<Graph>& graph) {
  size_t bin_size = -1;
  std::vector<char> nb_buf;
#ifdef VSI_DEVICE_SUPPORT
  GraphImpl* graphimp = dynamic_cast<GraphImpl*>(graph.get());
  vsi_nn_BindDevices(graphimp->graph(), 1, &sub_device_);
#endif
  auto ret = graph->CompileToBinary(nullptr, &bin_size);
  nb_buf.resize(bin_size);
  ret |= graph->CompileToBinary(nb_buf.data(), &bin_size);
  if(!ret) {
    VSILOGE("Compile fail");
    return nullptr;
  }

  std::shared_ptr<IExecutor> this_sp = shared_from_this();
  auto executable = std::make_shared<LiteNativeExecutableImpl>(this_sp, nb_buf);
  return executable;
}

LiteNativeExecutableImpl::LiteNativeExecutableImpl(
    const std::shared_ptr<IExecutor>& executor,
    const std::vector<char>& nb_buf) {
  executor_ = executor;
  context_ = nullptr;
  nb_graph_ = nullptr;
  vip_status_e  status = VIP_SUCCESS;
  vip_create_network_param_t net_param;
  device_id_ = executor_.lock()->Device()->Id();
  auto core_index  = executor_.lock()->CoreIndex();
  net_param.device_index = device_id_;
  net_param.prop = VIP_NET_CREATE_PROP_FROM_NBG;
  net_param.nbg.type = VIP_NET_CREATE_NBG_FROM_MEMORY;
  net_param.nbg.memory.nbg_memory = (void*)nb_buf.data();
  net_param.nbg.memory.nbg_size = nb_buf.size();

  auto network(std::make_unique<LiteNetwork>(net_param));

  lite_network_ = std::move(network);
  status = lite_network_->Query(VIP_NETWORK_PROP_INPUT_COUNT,&input_count_);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to query network inputs");
    assert(false);
  }
  status = lite_network_->Query(VIP_NETWORK_PROP_OUTPUT_COUNT,&output_count_);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to query network outputs");
    assert(false);
  }

  status = lite_network_->Set(VIP_NETWORK_PROP_SET_CORE_INDEX,&core_index);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to set core index");
    assert(false);
  }
   status = lite_network_->Prepare();
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to prepare network");
    assert(false);
  }
}

void LiteNativeExecutableImpl::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  vip_status_e status = VIP_SUCCESS;
  int32_t input_index = input_handles_.size();
  status = lite_network_->SetInput(input_index, th);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to set input: %d", input_index);
    assert(false);
  }
  input_handles_.push_back(th);
}
void LiteNativeExecutableImpl::SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) {
  for (auto th : ths) {
    SetInput(th);
  }
}

void LiteNativeExecutableImpl::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  vip_status_e status = VIP_SUCCESS;
  int32_t output_index = output_handles_.size();
  status = lite_network_->SetOutput(output_index,th);
  if (status != VIP_SUCCESS) {
    VSILOGE("failed to set output: %d", output_index);
    assert(false);
  }
  output_handles_.push_back(th);
}

void LiteNativeExecutableImpl::SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) {
  for (auto th : ths) {
    SetOutput(th);
  }
}

bool LiteNativeExecutableImpl::Submit(const std::shared_ptr<IExecutable>& ref,
                                  bool after) {
  bool status = false;
  std::shared_ptr<LiteNativeExecutorImpl> executor =
        std::dynamic_pointer_cast<LiteNativeExecutorImpl>(executor_.lock());
  std::shared_ptr<IExecutable> executable = shared_from_this();
  status = executor->Submit(executable, ref, after);
  return status;
}

bool LiteNativeExecutableImpl::Trigger(bool async) {
  vip_status_e status = VIP_SUCCESS;
  if (async) {
    status = lite_network_->Trigger();
    status = lite_network_->Wait();
    if (status != VIP_SUCCESS) {
      VSILOGE("trigger network fail");
      return false;
    }
  } else {
    status = lite_network_->Run();
    if (status != VIP_SUCCESS) {
      VSILOGE("run network fail");
      return false;
    }
  }
  return true;
}

bool LiteNativeExecutableImpl::Verify() {
  bool ret = true;
  auto output_index = output_handles_.size();
  auto input_index = input_handles_.size();
  if(input_index != input_count_) {
      VSILOGE("Network need %d inputs but gaving  %d.\n", input_count_, input_index);
      ret = false;
  }
  if(output_index != output_count_) {
     VSILOGE("Network need %d outputs but gaving  %d.\n", output_count_, output_index);
      ret = false;
  }

  return ret;
}

std::shared_ptr<ITensorHandle> LiteNativeExecutableImpl::AllocateTensor(const TensorSpec& tensor_spec,
                                                                        void* data, uint32_t size) {
  return std::make_shared<LiteNativeTensorHandleImpl>(tensor_spec, data, size, device_id_);
}

LiteNativeTensorHandleImpl::LiteNativeTensorHandleImpl(const TensorSpec& tensor_spec, void* data, uint32_t size,
                                                       uint32_t device_id) {
  vip_status_e  status  = VIP_ERROR_FAILURE;
  spec_  = tensor_spec;
  uint32_t tensor_size = tensor_spec.GetByteSize();
  vip_buffer_create_params_t tensor_param;

  uint32_t block_aligned_size = 64;
  memory_type_ = ALLOC_MEM_NONE;
  handle_ = nullptr;
  handle_size_ = 0;
  if(size > 0 && !data && tensor_size >  size ) {
    VSILOGE("Buffer size is less than the memory size required by the tensor");
    assert(false);
  }
#if 0
  uint32_t addr_aligned_size = 256;
  if (!data) {
    data = vsi_nn_MallocAlignedBuffer(tensor_size,addr_aligned_size,block_aligned_size);
    size = ((tensor_size + block_aligned_size - 1) / block_aligned_size) * block_aligned_size;
    memory_type_ = ALLOC_MEM_INTERNAL;
  } else {
    memory_type_ = ALLOC_MEM_EXTERNAL;
  }
  handle_ = data;
  if(!vsi_nn_IsBufferAligned((uint8_t *)handle_, addr_aligned_size)) {
      VSILOGE("The starting address of the buffer needs to be 64-byte aligned");
      assert(false);
  }
  if(size % 64 != 0) {
      VSILOGE("The size of the buffer needs to be 64-byte aligned");
      assert(false);
  }
  handle_size_ = size;
  tensor_param.type = VIP_BUFFER_CREATE_FROM_USER_MEM;
  tensor_param.device_index = device_id ;
  tensor_param.src.from_handle.memory_type = VIP_BUFFER_FROM_USER_MEM_TYPE_HOST;
  tensor_param.src.from_handle.logical_addr = handle_;
  tensor_param.src.from_handle.size = handle_size_;
  status = vip_create_buffer(&tensor_param,sizeof(tensor_param),&tensor_buffer_);
#else
  (void)data;
  tensor_param.type = VIP_BUFFER_CREATE_ALLOC_MEM;
  tensor_param.device_index = device_id ;
  tensor_param.src.alloc_mem.size = tensor_size;
  tensor_param.src.alloc_mem.align = block_aligned_size;
  status = vip_create_buffer(&tensor_param,sizeof(tensor_param),&tensor_buffer_);
  memory_type_ = ALLOC_MEM_VIDEOMEM;
#endif
  if(status != VIP_SUCCESS) {
    if(memory_type_ == ALLOC_MEM_INTERNAL) {
      vsi_nn_FreeAlignedBuffer((uint8_t*)handle_);
    }
    VSILOGE("Fail to create vip buffer.");
    assert(false);
  }
}

LiteNativeTensorHandleImpl::~LiteNativeTensorHandleImpl() {
  if (tensor_buffer_) {
    vip_destroy_buffer(tensor_buffer_);
    tensor_buffer_ = nullptr;
  }
  if(memory_type_ == ALLOC_MEM_INTERNAL && handle_) {
    vsi_nn_FreeAlignedBuffer((uint8_t*)handle_);

  }
}

bool LiteNativeTensorHandleImpl::CopyDataToTensor(const void* data,
                                                  uint32_t size_in_bytes) {
  void* handle  = handle_;
  if(memory_type_ == ALLOC_MEM_VIDEOMEM) {
    handle = vip_map_buffer(tensor_buffer_);
  }
  auto buff_size = vip_get_buffer_size(tensor_buffer_);
  memcpy(handle, data, buff_size > size_in_bytes ? size_in_bytes : buff_size);
  if(memory_type_ == ALLOC_MEM_VIDEOMEM) {
    vip_unmap_buffer(tensor_buffer_);
  }
  Flush();
  return true;
}

bool LiteNativeTensorHandleImpl::CopyDataFromTensor(void* data) {
  bool ret = Invalidate();
  if(ret) {
    void* handle  = handle_;
    auto buff_size = vip_get_buffer_size(tensor_buffer_);
    if(memory_type_ == ALLOC_MEM_VIDEOMEM) {
      handle = vip_map_buffer(tensor_buffer_);
    }
    memcpy(data, handle, buff_size);
    if(memory_type_ == ALLOC_MEM_VIDEOMEM) {
      vip_unmap_buffer(tensor_buffer_);
    }
  }

  return ret;
}

bool LiteNativeTensorHandleImpl::Flush() {
  vip_status_e status = vip_flush_buffer(tensor_buffer_,VIP_BUFFER_OPER_TYPE_FLUSH);
  if (status != VIP_SUCCESS) {
     return false;
  }
  else{
    return true;
  }
}
bool LiteNativeTensorHandleImpl::Invalidate() {
  vip_status_e status = vip_flush_buffer(tensor_buffer_,VIP_BUFFER_OPER_TYPE_INVALIDATE);
  if (status != VIP_SUCCESS) {
     return false;
  }
  else{
    return true;
  }
}

}  // namespace platform
}  // namespace vx
}  // namespace tim
