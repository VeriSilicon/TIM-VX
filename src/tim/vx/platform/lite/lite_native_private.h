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
#ifndef TIM_VX_LITE_NATIVE_DEVICE_PRIVATE_H_
#define TIM_VX_LITE_NATIVE_DEVICE_PRIVATE_H_

#include "tim/vx/platform/lite/lite_native.h"
#include "vip_lite.h"
#include "vsi_nn_pub.h"


namespace tim {
namespace vx {

namespace platform {

class LiteNetwork
{
public:
  LiteNetwork(vip_create_network_param_t& param);
  ~LiteNetwork();
  vip_status_e Query(vip_enum property, void* value);
  vip_status_e Set(vip_enum property, void* value);
  vip_status_e Prepare();
  vip_status_e Run();
  vip_status_e Trigger();
  vip_status_e Wait();
  vip_status_e Cancel();
  vip_status_e QueryInput(vip_uint32_t index, vip_enum property, void* value);
  vip_status_e QueryOutput(vip_uint32_t index, vip_enum property, void* value);
  vip_status_e SetInput(vip_uint32_t index, std::shared_ptr<ITensorHandle> input);
  vip_status_e SetOutput(vip_uint32_t index, std::shared_ptr<ITensorHandle> output);

private:
    vip_network network_;
};

class LiteNativeDeviceImpl : public LiteNativeDevice,
                             public std::enable_shared_from_this<LiteNativeDeviceImpl> {
 public:
  LiteNativeDeviceImpl(device_id_t id,uint32_t core_count);
  ~LiteNativeDeviceImpl() {};

  bool Submit(const std::shared_ptr<tim::vx::Graph>& graph) override;
  bool Trigger(bool async = false, async_callback cb = NULL) override;
  bool DeviceExit() override;
  void WaitDeviceIdle() override;
   std::shared_ptr<IExecutor> CreateExecutor(const int32_t core_index = 0,
                                             const int32_t core_count = -1,
                                             const std::shared_ptr<Context>& context = nullptr) override;
};

class LiteNativeExecutorImpl
    : public LiteNativeExecutor,
      public std::enable_shared_from_this<LiteNativeExecutorImpl> {
 public:
  LiteNativeExecutorImpl(const std::shared_ptr<IDevice>& device,
                     const int32_t core_index = 0,
                     const int32_t core_count = -1,
                     const std::shared_ptr<Context>& context = nullptr);
  virtual ~LiteNativeExecutorImpl();
  bool Submit(const std::shared_ptr<IExecutable>& executable,
              const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<IExecutable> Compile(const std::shared_ptr<Graph>& graph) override;
  static int executor_count;

private:
#ifdef VSI_DEVICE_SUPPORT
  vsi_nn_device_t  sub_device_;
#endif
};

class LiteNativeExecutableImpl : public LiteNativeExecutable {
 public:
  LiteNativeExecutableImpl(const std::shared_ptr<IExecutor>& executor,
                           const std::vector<char>& nb_buf);
  virtual ~LiteNativeExecutableImpl() {};
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetInputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) override;
  void SetOutputs(const std::vector<std::shared_ptr<ITensorHandle>>& ths) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref, bool after) override;
  bool Trigger(bool async) override;
  bool Verify() override;
  std::shared_ptr<ITensorHandle> AllocateTensor(const TensorSpec& tensor_spec,
                                                void* data = nullptr, uint32_t size = 0) override;

 private:
  uint32_t device_id_;
  uint32_t input_count_;
  uint32_t output_count_;
  std::unique_ptr<LiteNetwork> lite_network_;
};

class LiteNativeTensorHandleImpl : public LiteNativeTensorHandle {
 public:
   typedef enum {
    ALLOC_MEM_NONE,
    ALLOC_MEM_EXTERNAL,
    ALLOC_MEM_INTERNAL,
    ALLOC_MEM_VIDEOMEM,
    ALLOC_MEM_PHYSICAL,
    ALLOC_MEM_FD,
  } alloc_mem_type;

  LiteNativeTensorHandleImpl(const TensorSpec& tensor_spec,void* data, uint32_t size,uint32_t device_id);
  virtual ~LiteNativeTensorHandleImpl();
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) override;
  bool CopyDataFromTensor(void* data) override;
  bool Flush();
  bool Invalidate();
  vip_buffer GetBuffer() {return tensor_buffer_;};

private:
  vip_buffer tensor_buffer_;
  void* handle_;
  uint32_t handle_size_;
  alloc_mem_type memory_type_;
};

}  // namespace platform
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_NATIVE_DEVICE_PRIVATE_H_*/
