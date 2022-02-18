/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#ifndef TIM_VX_IDEVICE_H_
#define TIM_VX_IDEVICE_H_

#include <memory>
#include <vector>
#include <functional>
#include "graph.h"

namespace tim {
namespace vx {

class IDevice {
 public:
  using device_id_t = uint32_t;
  using async_callback = std::function<bool (const void*)>;
  using data_t = const void*;
  virtual ~IDevice(){
      printf("Destructor IDevice: %p\n", this);
  };
  virtual bool Submit(const std::shared_ptr<Graph> graph/*, func_t func=NULL, data_t data=NULL*/) = 0;
  virtual bool Trigger(bool async = false, async_callback cb = NULL) = 0;
  device_id_t device_id() {return device_id_;}
  virtual void WaitDeviceIdle() = 0;
  virtual bool DeviceExit() = 0;

 protected:
  device_id_t device_id_;
  
};


}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_IDEVICE_H_*/