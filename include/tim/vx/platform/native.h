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
#ifndef TIM_VX_NATIVE_H_
#define TIM_VX_NATIVE_H_

#include "tim/vx/platform/platform.h"

namespace tim {
namespace vx {
namespace platform {

class NativeDevice : public IDevice {
 public:
  ~NativeDevice(){};
  virtual bool Submit(const std::shared_ptr<Graph>& graph) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  virtual bool DeviceExit() = 0;
  virtual void WaitDeviceIdle() = 0;
  static std::vector<std::shared_ptr<IDevice>> Enumerate();
};

class NativeExecutable : public IExecutable {
 public:
  NativeExecutable(const std::shared_ptr<IExecutor>& executor,
                   const std::vector<char>& nb_buf, size_t inputs,
                   size_t outputs);
  ~NativeExecutable(){};
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void GetOutput(
      const std::vector<std::shared_ptr<ITensorHandle>>& th) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<ITensorHandle> AllocateTensor(
      const TensorSpec& tensor_spec) override;
  bool Verify() override;

 protected:
  std::shared_ptr<tim::vx::ops::NBG> nb_node_;
  std::vector<char> nb_buf_;
};

class NativeExecutor : public IExecutor,
                       public std::enable_shared_from_this<NativeExecutor> {
 public:
  NativeExecutor(const std::shared_ptr<IDevice>& device);
  NativeExecutor(const std::shared_ptr<IDevice>& device,
                 const std::shared_ptr<Context>& context);
  ~NativeExecutor(){};
  bool Submit(const std::shared_ptr<IExecutable>& executable,
              const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) override;
};

class NativeTensorHandle : public ITensorHandle {
 public:
  NativeTensorHandle(const std::shared_ptr<Tensor>& tensor);
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) override;
  bool CopyDataFromTensor(void* data) override;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim
#endif
