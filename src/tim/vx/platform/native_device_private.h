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
#ifndef TIM_VX_NATIVE_DEVICE_PRIVATE_H_
#define TIM_VX_NATIVE_DEVICE_PRIVATE_H_

#include "tim/vx/platform/native.h"
#include "vip/virtual_device.h"
#include "graph_private.h"

namespace tim {
namespace vx {

class GraphImpl;

namespace platform {

class NativeDeviceImpl : public NativeDevice {
 public:
  NativeDeviceImpl(device_id_t id);
  ~NativeDeviceImpl(){};

  bool Submit(const std::shared_ptr<tim::vx::Graph>& graph) override;
  bool Trigger(bool async = false, async_callback cb = NULL) override;
  bool DeviceExit() override;
  void WaitDeviceIdle() override;

 protected:
  std::unique_ptr<vip::IDevice> vip_device_;
  std::vector<vsi_nn_graph_t*> vsi_graph_v_;

};

}  // namespace platform
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_NATIVE_DEVICE_PRIVATE_H_*/