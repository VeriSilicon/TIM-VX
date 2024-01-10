/****************************************************************************
*
*    Copyright (c) 2020-2024 Vivante Corporation
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

#ifndef VSI_NBG_RUNNER_VX_UTILS_HPP_
#define VSI_NBG_RUNNER_VX_UTILS_HPP_

#include <VX/vx_types.h>

#include <string_view>

namespace vsi::nbg_runner::vx {

inline size_t get_vx_dtype_bytes(vx_enum data_type) {
  switch (data_type) {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
    case VX_TYPE_BOOL8:
    case VX_TYPE_CHAR:
      return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
    case VX_TYPE_FLOAT16:
    case VX_TYPE_BFLOAT16:
      return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
    case VX_TYPE_FLOAT32:
      return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
    case VX_TYPE_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

inline std::string_view get_vx_dtype_str(vx_enum data_type) {
  switch (data_type) {
    case VX_TYPE_INT8:
      return "int8";
    case VX_TYPE_UINT8:
      return "uint8";
    case VX_TYPE_BOOL8:
      return "bool";
    case VX_TYPE_INT16:
      return "int16";
    case VX_TYPE_UINT16:
      return "uint16";
    case VX_TYPE_FLOAT16:
      return "float16";
    case VX_TYPE_BFLOAT16:
      return "bfloat16";
    case VX_TYPE_INT32:
      return "int32";
    case VX_TYPE_UINT32:
      return "uint32";
    case VX_TYPE_FLOAT32:
      return "float32";
    case VX_TYPE_INT64:
      return "int64";
    case VX_TYPE_UINT64:
      return "uint64";
    case VX_TYPE_FLOAT64:
      return "float32";
    default:
      return "unknown";
  }
}

inline std::string_view get_vx_qtype_str(vx_enum quant_type) {
  switch (quant_type) {
    case VX_QUANT_DYNAMIC_FIXED_POINT:
      return "dfp";
    case VX_QUANT_AFFINE_SCALE:
      return "affine";
    case VX_QUANT_AFFINE_SCALE_PER_CHANNEL:
      return "perchannel_affine";
    case VX_QUANT_NONE:
    default:
      return "none";
  }
}

}  // namespace vsi::nbg_runner::vx
#endif