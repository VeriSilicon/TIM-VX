/****************************************************************************
*
*    Copyright (c) 2020-2025 Vivante Corporation
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
#ifndef TIM_VX_NATIVE_DEVICE_PRIVATE_H_
#define TIM_VX_NATIVE_DEVICE_PRIVATE_H_

#include "tim/vx/platform/native.h"
#include "vip/virtual_device.h"
#include "graph_private.h"

namespace tim {
namespace vx {

class GraphImpl;

namespace platform {

class NativeDeviceImpl : public NativeDevice,
                         public std::enable_shared_from_this<NativeDeviceImpl>{
 public:
  NativeDeviceImpl(device_id_t id,uint32_t core_count);
  ~NativeDeviceImpl(){};

  bool Submit(const std::shared_ptr<tim::vx::Graph>& graph) override;
  bool Trigger(bool async = false, async_callback cb = NULL) override;
  bool DeviceExit() override;
  void WaitDeviceIdle() override;
  std::shared_ptr<IExecutor> CreateExecutor(const int32_t core_index = 0,
                                            const int32_t core_count = -1,
                                            const std::shared_ptr<Context>& context = nullptr) override;
};

class NativeExecutableImpl : public NativeExecutable {
 public:
  NativeExecutableImpl(const std::shared_ptr<IExecutor>& executor,
                   const std::vector<char>& nb_buf, size_t inputs,
                   size_t outputs);
  ~NativeExecutableImpl() {};
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) override;
  void SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref, bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec,
                                                void* data = nullptr, uint32_t size = 0) override;
  bool Verify() override;

 protected:
  std::shared_ptr<tim::vx::ops::NBG> nb_node_;
  std::vector<char> nb_buf_;
};

class NativeExecutorImpl : public NativeExecutor,
                       public std::enable_shared_from_this<NativeExecutorImpl> {
 public:
  NativeExecutorImpl(const std::shared_ptr<IDevice>& device,
                 const int32_t core_count = -1,
                 const int32_t core_index = 0,
                 const std::shared_ptr<Context>& context = nullptr);
  ~NativeExecutorImpl(){};
  bool Submit(const std::shared_ptr<IExecutable>& executable,
              const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<IExecutable> Compile(const std::shared_ptr<Graph>& graph) override;
  bool BindDevices(const std::shared_ptr<Graph>& graph);

private:
#ifdef VSI_DEVICE_SUPPORT
  vsi_nn_device_t  sub_devices_;
#endif
};

class NativeTensorHandleImpl : public NativeTensorHandle {
 public:
  NativeTensorHandleImpl(const std::shared_ptr<Tensor>& tensor);
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) override;
  bool CopyDataFromTensor(void* data) override;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_NATIVE_DEVICE_PRIVATE_H_*/
