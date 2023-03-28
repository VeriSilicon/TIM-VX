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
#include "vip_lite.h"
#include "nbg_linker.h"

namespace tim {
namespace vx {
namespace platform {

class LiteNativeExecutor
    : public IExecutor,
      public std::enable_shared_from_this<LiteNativeExecutor> {
 public:
  LiteNativeExecutor(const std::shared_ptr<IDevice>& device);
  virtual ~LiteNativeExecutor();
  bool Submit(const std::shared_ptr<IExecutable>& executable,
              const std::shared_ptr<IExecutable>& ref,
              bool after = true) override;
  bool Trigger(bool async = false) override;
  std::shared_ptr<IExecutable> Compile(
      const std::shared_ptr<Graph>& graph) override;

 private:
  vip_task_descriptor_t* task_descriptor_;
  vip_database database_;
};

class LiteNativeExecutable : public IExecutable {
 public:
  LiteNativeExecutable(const std::shared_ptr<IExecutor>& executor,
                       const std::vector<char>& nb_buf);
  virtual ~LiteNativeExecutable();
  void SetInput(const std::shared_ptr<ITensorHandle>& th) override;
  void SetOutput(const std::shared_ptr<ITensorHandle>& th) override;
  void GetOutput(
      const std::vector<std::shared_ptr<ITensorHandle>>& th) override;
  bool Submit(const std::shared_ptr<IExecutable>& ref, bool after) override;
  bool Trigger(bool async) override;
  bool Verify() override;
  std::shared_ptr<ITensorHandle> AllocateTensor(
      const TensorSpec& tensor_spec) override;

  vip_network network_;

 private:
  void SetBuffer(vip_memory_t* dst, gcvip_videomemory_t* src);

  int32_t input_count_;
  int32_t output_count_;

  gcvip_videomemory_t* coeff_;
  gcvip_videomemory_t* command_;
  gcvip_videomemory_t* memory_pool_;
  gcvip_videomemory_t* others_;
  gcvip_videomemory_t* pre_command_;
};

class LiteNativeTensorHandle : public ITensorHandle {
 public:
  LiteNativeTensorHandle(const std::shared_ptr<Tensor>& tensr);
  virtual ~LiteNativeTensorHandle();
  bool CopyDataToTensor(const void* data, uint32_t size_in_bytes) override;
  bool CopyDataFromTensor(void* data) override;

  gcvip_videomemory_t* tensor_buffer_;
};
}  // namespace platform
}  // namespace vx
}  // namespace tim

#endif