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

#include "execution_private.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cassert>
#include "handle_private.h"

#include "vip_lite.h"

namespace tim {
namespace lite {

ExecutionImpl::ExecutionImpl(const void* executable, size_t executable_size) {
    vip_status_e status = VIP_SUCCESS;
    vip_network network = nullptr;
    std::vector<uint8_t> data(executable_size);
    valid_ = false;
    status = vip_init();
    if (status != VIP_SUCCESS) {
        return;
    }
    memcpy(data.data(), executable, executable_size);
    status = vip_create_network(data.data(), data.size(),
        VIP_CREATE_NETWORK_FROM_MEMORY, &network);
    if (status == VIP_SUCCESS && network) {
        status = vip_prepare_network(network);
        if (status == VIP_SUCCESS) {
            network_ = network;
            valid_ = true;
        } else {
            vip_destroy_network(network);
        }
    }
    if (!valid_) {
        vip_destroy();
    }
}

ExecutionImpl::~ExecutionImpl() {
    if (!valid_) {
        return;
    }
    if (network_) {
        vip_finish_network(network_);
        vip_destroy_network(network_);
    }
    input_handles_.clear();
    output_handles_.clear();
    vip_destroy();
}

std::shared_ptr<Handle> ExecutionImpl::CreateInputHandle(uint32_t in_idx, uint8_t* buffer, size_t size) {
    auto handle = std::make_shared<HandleImpl>(buffer, size);
    if (handle->CreateVipInputBuffer(network_, in_idx)) {
        return handle;
    } else {
        return nullptr;
    }
}

std::shared_ptr<Handle> ExecutionImpl::CreateOutputHandle(uint32_t out_idx, uint8_t* buffer, size_t size) {
    auto handle = std::make_shared<HandleImpl>(buffer, size);
    if (handle->CreateVipPOutputBuffer(network_, out_idx)) {
        return handle;
    } else {
        return nullptr;
    }
}

Execution& ExecutionImpl::BindInputs(const std::vector<std::shared_ptr<Handle>>& handles) {
    if (!IsValid()) {
        return *this;
    }
    for (auto handle : handles) {
        if (input_handles_.end() == std::find(input_handles_.begin(), input_handles_.end(), handle)) {
            input_handles_.push_back(handle);
            auto handle_impl = std::dynamic_pointer_cast<HandleImpl>(handle);
            vip_status_e status = vip_set_input(network_, handle_impl->Index(), handle_impl->VipHandle());
            if (status != VIP_SUCCESS) {
                std::cout << "Set input for network failed." << std::endl;
                assert(false);
            }
        } else {
            std::cout << "The input handle has been binded, need not bind it again." << std::endl;
        }
    }
    return *this;
};

Execution& ExecutionImpl::BindOutputs(const std::vector<std::shared_ptr<Handle>>& handles) {
    if (!IsValid()) {
        return *this;
    }
    for (auto handle : handles) {
        if (output_handles_.end() == std::find(output_handles_.begin(), output_handles_.end(), handle)) {
            output_handles_.push_back(handle);
            auto handle_impl = std::dynamic_pointer_cast<HandleImpl>(handle);
            vip_status_e status = vip_set_output(network_, handle_impl->Index(), handle_impl->VipHandle());
            if (status != VIP_SUCCESS) {
                std::cout << "Set output for network failed." << std::endl;
                assert(false);
            }
        } else {
            std::cout << "The output handle has been binded, need not bind it again." << std::endl;
        }
    }
    return *this;
};

Execution& ExecutionImpl::UnBindInput(const std::shared_ptr<Handle>& handle) {
    auto it = std::find(input_handles_.begin(), input_handles_.end(), handle);
    if (input_handles_.end() != it) {
        input_handles_.erase(it);
    }
    return *this;
}

Execution& ExecutionImpl::UnBindOutput(const std::shared_ptr<Handle>& handle) {
    auto it = std::find(output_handles_.begin(), output_handles_.end(), handle);
    if (output_handles_.end() != it) {
        output_handles_.erase(it);
    }
    return *this;
}

bool ExecutionImpl::Trigger() {
    if (!IsValid()) {
        return false;
    }
    vip_status_e status = vip_run_network(network_);
    return status == VIP_SUCCESS;
};

std::shared_ptr<Execution> Execution::Create(
    const void* executable, size_t executable_size) {
    std::shared_ptr<ExecutionImpl> exec;
    if (executable && executable_size) {
        exec = std::make_shared<ExecutionImpl>(executable, executable_size);
        if (!exec->IsValid()) {
            exec.reset();
        }
    }
    return exec;
}

}
}
