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

#include "handle_private.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <iostream>
#include <string.h>
#include "execution_private.h"
#include "vip_lite.h"

#define _64_BYTES_ALIGN      (64ul)

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

bool HandleImpl::CreateVipInputBuffer(vip_network network,
                                            uint32_t in_idx) {
  vip_status_e status = VIP_SUCCESS;
  vip_buffer_create_params_t param;
  vip_buffer internal_buffer;
  assert((reinterpret_cast<uintptr_t>(buffer_) % _64_BYTES_ALIGN) == 0);
  if (!QueryInputBufferParameters(param, in_idx, network)) {
    status = VIP_ERROR_FAILURE;
    return false;
  }
  status = vip_create_buffer_from_handle(&param, buffer_, buffer_size_,
                                         &internal_buffer);
  if (status == VIP_SUCCESS) {
    handle_ = internal_buffer;
    SetIndex(in_idx);
    return true;
  } else {
    handle_ = nullptr;
    return false;
  }
}

bool HandleImpl::CreateVipPOutputBuffer(vip_network network,
                                             uint32_t out_idx) {
  vip_status_e status = VIP_SUCCESS;
  vip_buffer_create_params_t param;
  vip_buffer internal_buffer;
  assert((reinterpret_cast<uintptr_t>(buffer_) % _64_BYTES_ALIGN) == 0);
  if (!QueryOutputBufferParameters(param, out_idx, network)) {
    status = VIP_ERROR_FAILURE;
    return false;
  }
  status = vip_create_buffer_from_handle(&param, buffer_, buffer_size_,
                                         &internal_buffer);
  if (status == VIP_SUCCESS) {
    handle_ = internal_buffer;
    SetIndex(out_idx);
    return true;
  } else {
    handle_ = nullptr;
    return false;
  }
}

bool HandleImpl::Flush() {
    vip_status_e status = vip_flush_buffer(handle_, VIP_BUFFER_OPER_TYPE_FLUSH);
    return status == VIP_SUCCESS ? true : false;
}

bool HandleImpl::Invalidate() {
    vip_status_e status = vip_flush_buffer(handle_, VIP_BUFFER_OPER_TYPE_INVALIDATE);
    return status == VIP_SUCCESS ? true : false;
}

HandleImpl::~HandleImpl() {
  if (handle_) {
    vip_destroy_buffer(handle_);
    handle_ = nullptr;
  }
}

}
}
