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
class IExecutor;
class ITensorHandle;

std::shared_ptr<IExecutable> Compile(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<IExecutor>& executor);

class IDevice {
 public:
  using device_id_t = uint32_t;
  #ifdef __ANDROID_NDK__
  typedef bool (*async_callback)(const void*);
  #else
  using async_callback = std::function<bool(const void*)>;
  #endif
  using data_t = const void*;
  virtual ~IDevice(){};
  virtual bool Submit(const std::shared_ptr<Graph>& graph) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  device_id_t Id() const { return device_id_;};
  virtual void WaitDeviceIdle() = 0;
  virtual bool DeviceExit() = 0;
  virtual void RemoteReset();
  uint32_t CoreCount() const {return core_count_;};
  virtual std::shared_ptr<IExecutor> CreateExecutor(const int32_t core_index = 0,
                                                    const int32_t core_count = -1,
                                                    const std::shared_ptr<Context>& context = nullptr) = 0;
  static std::vector<std::shared_ptr<IDevice>> Enumerate();

 protected:
  device_id_t device_id_;
  uint32_t core_count_;

};

class IExecutor {
 public:
  //using task = std::shared_ptr<IExecutable>;
  using task = std::weak_ptr<IExecutable>;
  virtual ~IExecutor(){};
  virtual bool Submit(const std::shared_ptr<IExecutable>& executable,
                      const std::shared_ptr<IExecutable>& ref,
                      bool after = true) = 0;
  virtual bool Trigger(bool async = false) = 0;  // todo: async=true
  virtual std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) = 0;
  virtual std::shared_ptr<IDevice> Device() const {return device_;};
  virtual std::shared_ptr<Context> Contex() const {return context_;};
  virtual uint32_t CoreIndex() const {return core_index_; };
  virtual uint32_t CoreCount() const {return core_count_; };
 protected:
  std::vector<task> tasks_;
  std::shared_ptr<IDevice> device_;
  std::shared_ptr<Context> context_;
  uint32_t core_index_;
  uint32_t core_count_;

};

class IExecutable : public std::enable_shared_from_this<IExecutable> {
 public:
  virtual ~IExecutable(){};
  virtual void SetInput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetOutput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) = 0;
  virtual void SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) = 0;
  virtual std::vector<std::shared_ptr<ITensorHandle>> GetOutputs() { return input_handles_;};
  virtual std::vector<std::shared_ptr<ITensorHandle>> Getinputs() { return input_handles_;};
  virtual bool Submit(const std::shared_ptr<IExecutable>& ref,
                      bool after = true) = 0;
  virtual bool Trigger(bool async = false) = 0;  // todo: async=true
  virtual bool Verify() = 0;
  std::shared_ptr<Graph> NBGraph() const {return nb_graph_;};
  virtual std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec ,
                                                        void* data = nullptr, uint32_t size = 0) = 0;

 protected:
  std::weak_ptr<IExecutor> executor_;
  std::shared_ptr<Context> context_;
  std::shared_ptr<Graph> nb_graph_;
  std::vector<std::shared_ptr<ITensorHandle>> input_handles_;
  std::vector<std::shared_ptr<ITensorHandle>> output_handles_;
};

class ITensorHandle {
 public:
  virtual ~ITensorHandle(){};
  virtual bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) = 0;
  virtual bool CopyDataFromTensor(void* data) = 0;
  virtual std::shared_ptr<Tensor> GetTensor() const { return tensor_;};
  virtual TensorSpec& GetSpec() { return spec_;};

 protected:
  std::shared_ptr<Tensor> tensor_;
  TensorSpec spec_;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim
#endif
