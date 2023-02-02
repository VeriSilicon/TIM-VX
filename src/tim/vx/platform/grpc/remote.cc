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
#include "tim/vx/platform/grpc/remote.h"
#include "tim/vx/platform/platform.h"
#include "remote_service_client.h"

namespace tim {
namespace vx {
namespace platform {

std::vector<std::shared_ptr<IDevice>> RemoteDevice::Enumerate(
    const std::string& port) {
  auto client = std::make_shared<RemoteServiceClient>(port);
  int32_t count = client->Enumerate();
  std::vector<std::shared_ptr<IDevice>> devices;
  for (int i = 0; i < count; ++i) {
    devices.push_back(std::make_shared<RemoteDevice>(i, client));
  }
  return devices;
}

RemoteDevice::RemoteDevice(int32_t id,
                           std::shared_ptr<RemoteServiceClient> client)
    : client_(client) {
  device_id_ = id;
}

RemoteDevice::~RemoteDevice() {}

bool RemoteDevice::Submit(const std::shared_ptr<Graph>& graph) {
  (void)graph;
  return false;
}

bool RemoteDevice::Trigger(bool async, async_callback cb) {
  (void)async;
  (void)cb;
  return false;
}

bool RemoteDevice::DeviceExit() { return false; }

void RemoteDevice::WaitDeviceIdle() {}

void RemoteDevice::RemoteReset() { client_->Clean(); }

RemoteExecutor::RemoteExecutor(std::shared_ptr<IDevice> device)
    : device_(device) {
  executor_id_ =
      std::dynamic_pointer_cast<RemoteDevice>(device)->client_->CreateExecutor(
          device->Id());
}

bool RemoteExecutor::Submit(const std::shared_ptr<IExecutable>& executable,
                            const std::shared_ptr<IExecutable>& ref,
                            bool after) {
  (void)executable;
  (void)ref;
  (void)after;
  return false;
}

bool RemoteExecutor::Trigger(bool async) {
  (void)async;
  return std::dynamic_pointer_cast<RemoteDevice>(device_)->client_->Trigger(
      executor_id_);
}

std::shared_ptr<IExecutable> RemoteExecutor::Compile(
    const std::shared_ptr<Graph>& graph) {
  size_t inputs_num = graph->InputsTensor().size();
  size_t outputs_num = graph->OutputsTensor().size();
  size_t nbg_size = -1;

  graph->CompileToBinary(nullptr, &nbg_size);
  std::vector<char> nbg_buf(nbg_size);
  graph->CompileToBinary(nbg_buf.data(), &nbg_size);

  int32_t executable_id =
      std::dynamic_pointer_cast<RemoteDevice>(device_)
          ->client_->CreateExecutable(executor_id_, nbg_buf, inputs_num,
                                      outputs_num);

  return std::make_shared<RemoteExecutable>(executable_id, device_);
}

int32_t RemoteExecutor::Id() const { return executor_id_; }

RemoteExecutable::RemoteExecutable(int32_t id, std::shared_ptr<IDevice> device)
    : executable_id_(id), device_(device) {}

void RemoteExecutable::SetInput(const std::shared_ptr<ITensorHandle>& th) {
  int32_t tensor_id = std::dynamic_pointer_cast<RemoteTensorHandle>(th)->Id();
  std::dynamic_pointer_cast<RemoteDevice>(device_)->client_->SetInput(
      executable_id_, tensor_id);
}

void RemoteExecutable::SetOutput(const std::shared_ptr<ITensorHandle>& th) {
  int32_t tensor_id = std::dynamic_pointer_cast<RemoteTensorHandle>(th)->Id();
  std::dynamic_pointer_cast<RemoteDevice>(device_)->client_->SetOutput(
      executable_id_, tensor_id);
}

void RemoteExecutable::GetOutput(
    const std::vector<std::shared_ptr<ITensorHandle>>& th) {
  (void)th;
}

bool RemoteExecutable::Submit(const std::shared_ptr<IExecutable>& ref,
                              bool after) {
  (void)after;
  int32_t executable_id =
      std::dynamic_pointer_cast<RemoteExecutable>(ref)->Id();
  return std::dynamic_pointer_cast<RemoteDevice>(device_)->client_->Submit(
      executable_id);
}

bool RemoteExecutable::Trigger(bool async) {
  (void)async;
  return false;
}

bool RemoteExecutable::Verify() { return false; }

std::shared_ptr<ITensorHandle> RemoteExecutable::AllocateTensor(
    const TensorSpec& tensor_spec) {
  int32_t tensor_id =
      std::dynamic_pointer_cast<RemoteDevice>(device_)->client_->AllocateTensor(
          executable_id_, tensor_spec);

  return std::make_shared<RemoteTensorHandle>(tensor_id, device_);
}

int32_t RemoteExecutable::Id() const { return executable_id_; }

RemoteTensorHandle::RemoteTensorHandle(int32_t id,
                                       std::shared_ptr<IDevice> device)
    : tensor_id_(id), device_(device) {}

bool RemoteTensorHandle::CopyDataToTensor(const void* data,
                                          uint32_t size_in_bytes) {
  return std::dynamic_pointer_cast<RemoteDevice>(device_)
      ->client_->CopyDataToTensor(tensor_id_, data, size_in_bytes);
}

bool RemoteTensorHandle::CopyDataFromTensor(void* data) {
  return std::dynamic_pointer_cast<RemoteDevice>(device_)
      ->client_->CopyDataFromTensor(tensor_id_, data);
}

int32_t RemoteTensorHandle::Id() const { return tensor_id_; }
}  // namespace platform
}  // namespace vx
}  // namespace tim
