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
#include "tim/vx/compile_option.h"
#include <string>

namespace tim {
namespace vx {

CompileOption CompileOption::DefaultOptions;

struct CompileOptionImpl {
  // string: readable name; bool: setup or not; bool: value if setup; bool: default value if not setup;
  using RelaxModeType = std::tuple<std::string, bool, bool, bool>;
  CompileOptionImpl() {
    relax_mode_ = RelaxModeType(std::string("RelaxMode"), false, false, false);
    #if defined(ENABLE_PLATFORM)
    device_id_ = 0;
    #endif
  }

  bool RelaxMode() const {
    return std::get<1>(relax_mode_) ? std::get<2>(relax_mode_)
                                    : std::get<3>(relax_mode_);
  }

  bool& RelaxMode() {
    return std::get<1>(relax_mode_) ? std::get<2>(relax_mode_)
                                    : std::get<3>(relax_mode_);
  }

#if defined(ENABLE_PLATFORM)
  void setDeviceId(::tim::vx::platform::IDevice::device_id_t device) {
    device_id_ = device;
  }
  ::tim::vx::platform::IDevice::device_id_t getDeviceId() {
    return device_id_;
  }

  ::tim::vx::platform::IDevice::device_id_t device_id_;
#endif

  RelaxModeType relax_mode_;
};

CompileOption::CompileOption() : impl_(new CompileOptionImpl()) {}

bool CompileOption::isRelaxMode() const { return this->impl_->RelaxMode(); }

bool CompileOption::setRelaxMode(bool enable) {
  return this->impl_->RelaxMode() = enable;
}

#if defined(ENABLE_PLATFORM)
  void CompileOption::setDeviceId(::tim::vx::platform::IDevice::device_id_t device) {
    this->impl_->setDeviceId(device);
  }

  ::tim::vx::platform::IDevice::device_id_t CompileOption::getDeviceId() {
    return this->impl_->getDeviceId();
  }


#endif
}  // namespace vx
}  // namespace tim
