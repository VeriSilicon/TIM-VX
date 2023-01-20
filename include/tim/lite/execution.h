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

#ifndef TIM_LITE_EXECUTION_H_
#define TIM_LITE_EXECUTION_H_

#include <cstdint>
#include <vector>
#include <memory>
#include "tim/lite/handle.h"

namespace tim {
namespace lite {

class Execution {
 public:
  static std::shared_ptr<Execution> Create(const void* executable,
                                           size_t executable_size);
  virtual std::shared_ptr<Handle> CreateInputHandle(uint32_t in_idx,
                                                    uint8_t* buffer,
                                                    size_t size) = 0;
  virtual std::shared_ptr<Handle> CreateOutputHandle(uint32_t out_idx,
                                                     uint8_t* buffer,
                                                     size_t size) = 0;
  virtual Execution& BindInputs(
      const std::vector<std::shared_ptr<Handle>>& handles) = 0;
  virtual Execution& BindOutputs(
      const std::vector<std::shared_ptr<Handle>>& handles) = 0;
  virtual Execution& UnBindInput(const std::shared_ptr<Handle>& Handle) = 0;
  virtual Execution& UnBindOutput(const std::shared_ptr<Handle>& handle) = 0;
  virtual bool Trigger() = 0;
};

}  // namespace lite
}  // namespace tim
#endif