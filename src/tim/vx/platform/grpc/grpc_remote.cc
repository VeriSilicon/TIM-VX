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
#include "tim/vx/platform/grpc/grpc_remote.h"

#include "tim/vx/platform/platform.h"
#include "grpc_platform_client.h"

namespace tim {
namespace vx {
namespace platform {

std::vector<std::shared_ptr<IDevice>> GRPCRemoteDevice::Enumerate(
    const std::string& port) {
  auto client = std::make_shared<GRPCPlatformClient>(port);
  int32_t count = client->Enumerate();
  std::vector<std::shared_ptr<IDevice>> devices;
  for (int i = 0; i < count; ++i) {
    devices.push_back(std::make_shared<GRPCRemoteDevice>(i, client));
  }
  return devices;
}

GRPCRemoteDevice::GRPCRemoteDevice(int32_t id,
                                   std::shared_ptr<GRPCPlatformClient> client)
    : client_(client) {
  device_id_ = id;
}

bool GRPCRemoteDevice::Submit(const std::shared_ptr<Graph>& graph) {
  (void)graph;
  return false;
}

bool GRPCRemoteDevice::Trigger(bool async, async_callback cb) {
  (void)async;
  (void)cb;
  return false;
}

bool GRPCRemoteDevice::DeviceExit() { return false; }

void GRPCRemoteDevice::WaitDeviceIdle() {}

void GRPCRemoteDevice::RemoteReset() { client_->Clean(); }

GRPCRemoteExecutor::GRPCRemoteExecutor(std::shared_ptr<IDevice> device)
    : device_(device) {
  executor_id_ = std::dynamic_pointer_cast<GRPCRemoteDevice>(device)
                     ->client_->CreateExecutor(device->Id());
}

bool GRPCRemoteExecutor::Submit(const std::shared_ptr<IExecutable>& executable,
                                const std::shared_ptr<IExecutable>& ref,
                                bool after) {
  (void)executable;
  (void)ref;
  (void)after;
  return false;
}

bool GRPCRemoteExecutor::Trigger(bool async) {
  (void)async;
  return std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)->client_->Trigger(
      executor_id_);
}

std::shared_ptr<IExecutable> GRPCRemoteExecutor::Compile(
    const std::shared_ptr<Graph>& graph) {
  size_t inputs_num = graph->InputsTensor().size();
  size_t outputs_num = graph->OutputsTensor().size();
  size_t nbg_size = -1;

  graph->CompileToBinary(nullptr, &nbg_size);
  std::vector<char> nbg_buf(nbg_size);
  graph->CompileToBinary(nbg_buf.data(), &nbg_size);

  int32_t executable_id =
      std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)
          ->client_->CreateExecutable(executor_id_, nbg_buf, inputs_num,
                                      outputs_num);

  return std::make_shared<GRPCRemoteExecutable>(executable_id, device_);
}

int32_t GRPCRemoteExecutor::Id() const { return executor_id_; }

GRPCRemoteExecutable::GRPCRemoteExecutable(int32_t id,
                                           std::shared_ptr<IDevice> device)
    : executable_id_(id), device_(device) {}

void GRPCRemoteExecutable::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  int32_t tensor_id =
      std::dynamic_pointer_cast<GRPCRemoteTensorHandle>(th)->Id();
  std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)->client_->SetInput(
      executable_id_, tensor_id);
}

void GRPCRemoteExecutable::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  int32_t tensor_id =
      std::dynamic_pointer_cast<GRPCRemoteTensorHandle>(th)->Id();
  std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)->client_->SetOutput(
      executable_id_, tensor_id);
}

void GRPCRemoteExecutable::GetOutput(
    const std::vector<std::shared_ptr<ITensorHandle>>& th) {
  (void)th;
}

bool GRPCRemoteExecutable::Submit(const std::shared_ptr<IExecutable>& ref,
                                  bool after) {
  (void)after;
  int32_t executable_id =
      std::dynamic_pointer_cast<GRPCRemoteExecutable>(ref)->Id();
  return std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)->client_->Submit(
      executable_id);
}

bool GRPCRemoteExecutable::Trigger(bool async) {
  (void)async;
  return false;
}

bool GRPCRemoteExecutable::Verify() { return false; }

std::shared_ptr<ITensorHandle> GRPCRemoteExecutable::AllocateTensor(
    const TensorSpec& tensor_spec) {
  int32_t tensor_id =
      std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)
          ->client_->AllocateTensor(executable_id_, tensor_spec);

  return std::make_shared<GRPCRemoteTensorHandle>(tensor_id, device_);
}

int32_t GRPCRemoteExecutable::Id() const { return executable_id_; }

GRPCRemoteTensorHandle::GRPCRemoteTensorHandle(int32_t id,
                                               std::shared_ptr<IDevice> device)
    : tensor_id_(id), device_(device) {}

bool GRPCRemoteTensorHandle::CopyDataToTensor(const void* data,
                                              uint32_t size_in_bytes) {
  return std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)
      ->client_->CopyDataToTensor(tensor_id_, data, size_in_bytes);
}

bool GRPCRemoteTensorHandle::CopyDataFromTensor(void* data) {
  return std::dynamic_pointer_cast<GRPCRemoteDevice>(device_)
      ->client_->CopyDataFromTensor(tensor_id_, data);
}

int32_t GRPCRemoteTensorHandle::Id() const { return tensor_id_; }
}  // namespace platform
}  // namespace vx
}  // namespace tim
