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
#ifndef TIM_VX_PLATFORM_H_
#define TIM_VX_PLATFORM_H_

#include <memory>
#include <vector>
#include <functional>
#include <iostream>
#include "tim/vx/graph.h"
#include "tim/vx/tensor.h"
#include "tim/vx/context.h"

namespace tim {
namespace vx {

class Graph;
class Context;

namespace ops {
class NBG;
}

namespace platform {

class IDevice;
class IExecutable;
class ExecutableSet;
class IExecutor;
class ITensorHandle;

std::shared_ptr<IExecutable> Compile(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<IExecutor>& executor);
std::shared_ptr<IExecutable> CreateExecutableSet(
    const std::vector<std::shared_ptr<IExecutable>>& executables);

class IDevice {
 public:
  using device_id_t = uint32_t;
  using async_callback = std::function<bool(const void*)>;
  using data_t = const void*;
  virtual ~IDevice(){};
  virtual bool Submit(const std::shared_ptr<Graph>& graph) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  device_id_t Id() const;
  virtual void WaitDeviceIdle() = 0;
  virtual bool DeviceExit() = 0;
  virtual void RemoteReset();

 protected:
  device_id_t device_id_;
};

class IExecutor {
 public:
  using task = std::weak_ptr<IExecutable>;
  virtual ~IExecutor(){};
  virtual bool Submit(const std::shared_ptr<IExecutable>& executable,
                      const std::shared_ptr<IExecutable>& ref,
                      bool after = true) = 0;
  virtual bool Trigger(bool async = false) = 0;  // todo: async=true
  virtual std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) = 0;
  virtual std::shared_ptr<IDevice> Device() const;
  virtual std::shared_ptr<Context> Contex() const;

 protected:
  std::vector<task> tasks_;
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<Context> context_;
};

class IExecutable : public std::enable_shared_from_this<IExecutable> {
 public:
  virtual ~IExecutable(){};
  virtual void SetInput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetOutput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void GetOutput(
      const std::vector<std::shared_ptr<ITensorHandle>>& th) = 0;  // for remote
  virtual bool Submit(const std::shared_ptr<IExecutable>& ref,
                      bool after = true) = 0;
  virtual bool Trigger(bool async = false) = 0;  // todo: async=true
  virtual bool Verify() = 0;
  virtual std::shared_ptr<Graph> NBGraph() const;
  virtual std::shared_ptr<ITensorHandle> AllocateTensor(
      const TensorSpec& tensor_spec) = 0;
  virtual std::shared_ptr<IExecutor> Executor() const;

 protected:
  std::weak_ptr<IExecutor> executor_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<Graph> nb_graph_;
};

class ExecutableSet : public IExecutable {
 public:
  ExecutableSet(const std::vector<std::shared_ptr<IExecutable>>& executables);
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void GetOutput(
      const std::vector<std::shared_ptr<ITensorHandle>>& th) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  bool Verify() override;
  std::shared_ptr<ITensorHandle> AllocateTensor(
      const TensorSpec& tensor_spec) override;
  std::vector<std::shared_ptr<IExecutable>> Executables() const;

 protected:
  std::vector<std::shared_ptr<IExecutable>> executables_;
};

class ITensorHandle {
 public:
  virtual ~ITensorHandle(){};
  virtual bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) = 0;
  virtual bool CopyDataFromTensor(void* data) = 0;
  virtual std::shared_ptr<Tensor> GetTensor() const;

 protected:
  std::shared_ptr<Tensor> tensor_;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim
#endif