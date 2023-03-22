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
#ifndef TIM_VX_GRPC_REMOTE_H_
#define TIM_VX_GRPC_REMOTE_H_

#include "tim/vx/platform/platform.h"

namespace tim {
namespace vx {
namespace platform {

class GRPCPlatformClient;

class GRPCRemoteDevice : public IDevice {
 public:
  GRPCRemoteDevice(int32_t id, std::shared_ptr<GRPCPlatformClient> client);
  bool Submit(const std::shared_ptr<Graph>& graph) override;
  bool Trigger(bool async = false, async_callback cb = NULL) override;
  bool DeviceExit() override;
  void WaitDeviceIdle() override;
  void RemoteReset() override;
  static std::vector<std::shared_ptr<IDevice>> Enumerate(
      const std::string& port);

  std::shared_ptr<GRPCPlatformClient> client_;
};

class GRPCRemoteExecutor : public IExecutor {
 public:
  GRPCRemoteExecutor(std::shared_ptr<IDevice> device);
  bool Submit(const std::shared_ptr<IExecutable>& executable,
              const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) override;
  int32_t Id() const;

 private:
  int32_t executor_id_;
  std::shared_ptr<IDevice> device_;
};

class GRPCRemoteExecutable : public IExecutable {
 public:
  GRPCRemoteExecutable(int32_t id, std::shared_ptr<IDevice> device);
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void GetOutput(
      const std::vector<std::shared_ptr<ITensorHandle>>& th) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref, bool after) override;
  bool Trigger(bool async) override;
  bool Verify() override;
  std::shared_ptr<ITensorHandle> AllocateTensor(
      const TensorSpec& tensor_spec) override;
  int32_t Id() const;

 private:
  int32_t executable_id_;
  std::shared_ptr<IDevice> device_;
};

class GRPCRemoteTensorHandle : public ITensorHandle {
 public:
  GRPCRemoteTensorHandle(int32_t id, std::shared_ptr<IDevice> device);
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) override;
  bool CopyDataFromTensor(void* data) override;
  int32_t Id() const;

 private:
  int32_t tensor_id_;
  std::shared_ptr<IDevice> device_;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim
#endif
