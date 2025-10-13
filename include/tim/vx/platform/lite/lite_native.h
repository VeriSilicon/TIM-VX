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
#ifndef TIM_VX_LITE_NATIVE_H_
#define TIM_VX_LITE_NATIVE_H_

#include "tim/vx/platform/platform.h"

namespace tim {
namespace vx {
namespace platform {

class LiteNativeDevice : public IDevice {
 public:
  virtual ~LiteNativeDevice() {};
  virtual bool Submit(const std::shared_ptr<Graph>& graph) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  virtual bool DeviceExit() = 0;
  virtual void WaitDeviceIdle() = 0;
  virtual std::shared_ptr<IExecutor> CreateExecutor(const int32_t core_index = 0,
                                                    const int32_t core_count = -1,
                                                    const std::shared_ptr<Context>& context = nullptr) = 0;
  static std::vector<std::shared_ptr<IDevice>> Enumerate();
  static bool vip_initialized;
};
class LiteNativeExecutor
    : public IExecutor {
 public:
  virtual ~LiteNativeExecutor() {};
  virtual bool Submit(const std::shared_ptr<IExecutable>& executable,
                      const std::shared_ptr<IExecutable>& ref,
                      bool after = true) = 0;
  virtual bool Trigger(bool async = false) = 0;
  virtual std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) = 0;
};

class LiteNativeExecutable : public IExecutable {
 public:
  virtual ~LiteNativeExecutable() {};
  virtual void SetInput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetOutput(const std::shared_ptr<ITensorHandle>& th) = 0;
  virtual void SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) = 0;
  virtual void SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) = 0;
  virtual bool Submit(const std::shared_ptr<IExecutable>& ref, bool after) = 0;
  virtual bool Trigger(bool async) = 0;
  virtual bool Verify() = 0;
  virtual std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec,
                                                        void* data = nullptr, uint32_t size = 0) = 0;
};

class LiteNativeTensorHandle : public ITensorHandle {
 public:
  virtual ~LiteNativeTensorHandle() {};
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) = 0;
  bool CopyDataFromTensor(void* data) = 0;
};
}  // namespace platform
}  // namespace vx
}  // namespace tim

#endif
