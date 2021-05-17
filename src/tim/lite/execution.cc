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

#include "execution_private.h"

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include "handle_private.h"

#include "vip_lite.h"

namespace tim {
namespace lite {

namespace {
bool QueryInputBufferParameters(
    vip_buffer_create_params_t& param, uint32_t index, vip_network network) {
    uint32_t count = 0;
    vip_query_network(network, VIP_NETWORK_PROP_INPUT_COUNT, &count);
    if (index >= count) {
        return false;
    }
    memset(&param, 0, sizeof(param));
    param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_query_input(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
    vip_query_input(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
    vip_query_input(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
    vip_query_input(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
    switch(param.quant_format) {
        case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
            vip_query_input(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                            &param.quant_data.dfp.fixed_point_pos);
            break;
        case VIP_BUFFER_QUANTIZE_TF_ASYMM:
            vip_query_input(network, index, VIP_BUFFER_PROP_TF_SCALE,
                            &param.quant_data.affine.scale);
            vip_query_input(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                            &param.quant_data.affine.zeroPoint);
        default:
            break;
    }
    return true;
}

bool QueryOutputBufferParameters(
    vip_buffer_create_params_t& param, uint32_t index, vip_network network) {
    uint32_t count = 0;
    vip_query_network(network, VIP_NETWORK_PROP_OUTPUT_COUNT, &count);
    if (index >= count) {
        return false;
    }
    memset(&param, 0, sizeof(param));
    param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
    vip_query_output(network, index, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
    vip_query_output(network, index, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
    vip_query_output(network, index, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
    vip_query_output(network, index, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
    switch(param.quant_format) {
        case VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
            vip_query_output(network, index, VIP_BUFFER_PROP_FIXED_POINT_POS,
                             &param.quant_data.dfp.fixed_point_pos);
            break;
        case VIP_BUFFER_QUANTIZE_TF_ASYMM:
            vip_query_output(network, index, VIP_BUFFER_PROP_TF_SCALE,
                             &param.quant_data.affine.scale);
            vip_query_output(network, index, VIP_BUFFER_PROP_TF_ZERO_POINT,
                             &param.quant_data.affine.zeroPoint);
            break;
        default:
        break;
    }
    return true;
}
}

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
    input_maps_.clear();
    output_maps_.clear();
    vip_destroy();
}

Execution& ExecutionImpl::BindInputs(const std::vector<std::shared_ptr<Handle>>& handles) {
    if (!IsValid()) {
        return *this;
    }
    vip_status_e status = VIP_SUCCESS;
    vip_buffer_create_params_t param = { 0 };
    for (uint32_t i = 0; i < handles.size(); i ++) {
        auto handle = handles[i];
        if (!handle) {
            status = VIP_ERROR_FAILURE;
            break;
        }
        std::shared_ptr<InternalHandle> internal_handle = nullptr;
        if (input_maps_.find(handle) == input_maps_.end()) {
            if (!QueryInputBufferParameters(param, i, network_)) {
                status = VIP_ERROR_FAILURE;
                break;
            }
            internal_handle = handle->impl()->Register(param);
            if (!internal_handle) {
                status = VIP_ERROR_FAILURE;
                break;
            }
            input_maps_[handle] = internal_handle;
        } else {
            internal_handle = input_maps_.at(handle);
        }
        status = vip_set_input(network_, i, internal_handle->handle());
        if (status != VIP_SUCCESS) {
            break;
        }
    }
    return *this;
};

Execution& ExecutionImpl::BindOutputs(const std::vector<std::shared_ptr<Handle>>& handles) {
    if (!IsValid()) {
        return *this;
    }
    vip_status_e status = VIP_SUCCESS;
    vip_buffer_create_params_t param = { 0 };
    for (uint32_t i = 0; i < handles.size(); i ++) {
        auto handle = handles[i];
        if (!handle) {
            status = VIP_ERROR_FAILURE;
            break;
        }
        std::shared_ptr<InternalHandle> internal_handle = nullptr;
        if (output_maps_.find(handle) == output_maps_.end()) {
            if (!QueryOutputBufferParameters(param, i, network_)) {
                status = VIP_ERROR_FAILURE;
                break;
            }
            internal_handle = handle->impl()->Register(param);
            if (!internal_handle) {
                status = VIP_ERROR_FAILURE;
                break;
            }
            output_maps_[handle] = internal_handle;
        } else {
            internal_handle = output_maps_.at(handle);
        }
        status = vip_set_output(network_, i, internal_handle->handle());
        if (status != VIP_SUCCESS) {
            break;
        }
    }
    return *this;
};

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
