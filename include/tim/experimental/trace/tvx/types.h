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
#ifndef TIM_EXPERIMENTAL_TRACE_TVX_TYPES_H_
#define TIM_EXPERIMENTAL_TRACE_TVX_TYPES_H_
#include "tim/vx/types.h"

namespace trace {

namespace target = ::tim::vx;

using ShapeType = std::vector<uint32_t>;

using DataType = target::DataType;

using QuantType = target::QuantType;

using TensorAttribute = target::TensorAttribute;

using PadType = target::PadType;

using PoolType = target::PoolType;

using RoundType = target::RoundType;

using OverflowPolicy = target::OverflowPolicy;

using RoundingPolicy = target::RoundingPolicy;

using ResizeType = target::ResizeType;

using DataLayout = target::DataLayout;

} /* namespace trace */

#endif // TIM_EXPERIMENTAL_TRACE_TVX_TYPES_H_
