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
#ifndef TIM_VX_COMPILE_OPTION_H_
#define TIM_VX_COMPILE_OPTION_H_

#include <map>
#include <memory>

#if defined(ENABLE_PLATFORM)
#include "platform/platform.h"
#endif

namespace tim {
namespace vx {
struct CompileOptionImpl;
class CompileOption {
 public:
  CompileOption();
  ~CompileOption(){};

  bool isRelaxMode() const;
  bool setRelaxMode(bool enable = false);

#if defined(ENABLE_PLATFORM)
  void setDeviceId(::tim::vx::platform::IDevice::device_id_t device);
  ::tim::vx::platform::IDevice::device_id_t getDeviceId();
#endif

  static CompileOption DefaultOptions;

 private:
  // option can have dafult values
  std::shared_ptr<CompileOptionImpl> impl_;
};
}  // namespace vx
}  // namespace tim

#endif
