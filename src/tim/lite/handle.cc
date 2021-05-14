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

#include "handle_private.h"

#include <cassert>
#include <cstdint>
#include "execution_private.h"
#include "vip_lite.h"

#define _64_BYTES_ALIGN      (64ul)

namespace tim {
namespace lite {

UserHandle::UserHandle(void* buffer, size_t size) {
    assert((reinterpret_cast<uintptr_t>(buffer) % _64_BYTES_ALIGN) == 0);
    impl_ = std::make_unique<UserHandleImpl>(buffer, size);
} 

UserHandle::~UserHandle() {
    impl()->Unregister();
}

bool UserHandleImpl::Register(vip_buffer_create_params_t& params) {
    bool ret = true;
    std::call_once(register_once_, [&ret, &params, this]() {
        vip_status_e status = VIP_SUCCESS;
        vip_buffer internal_buffer = nullptr;
        status = vip_create_buffer_from_handle(&params,
            this->user_buffer(),
            this->user_buffer_size(),
            &internal_buffer);
            if (status == VIP_SUCCESS) {
                this->handle_ = internal_buffer;
                ret = true;
            } else {
                ret = false;
            }
    });
    return ret;
}

bool UserHandleImpl::Unregister() {
    if (handle_) {
        vip_destroy_buffer(handle_);
        handle_ = nullptr;
    }
    return true;
}

}
}
