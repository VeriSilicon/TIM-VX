/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#ifndef TIM_LITE_EXECUTION_PRIVATE_H_
#define TIM_LITE_EXECUTION_PRIVATE_H_

#include <vector>
#include <memory>
#include <map>

#include "tim/lite/execution.h"
#include "handle_private.h"
#include "vip_lite.h"

namespace tim {
namespace lite {

class ExecutionImpl : public Execution {
    public :
        ExecutionImpl(const void* executable, size_t executable_size);
        ~ExecutionImpl();
        Execution& BindInputs(const std::vector<std::shared_ptr<Handle>>& handles) override;
        Execution& BindOutputs(const std::vector<std::shared_ptr<Handle>>& handles) override;
        bool Trigger() override;
        bool IsValid() const { return valid_; };
        vip_network network() { return network_; };
    private:
        std::map<std::shared_ptr<Handle>, std::shared_ptr<InternalHandle>> input_maps_;
        std::map<std::shared_ptr<Handle>, std::shared_ptr<InternalHandle>> output_maps_;
        bool valid_;
        vip_network network_;
};

}
}
#endif