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
#include "op_impl.h"

namespace tim {
namespace vx {

OpImpl::OpImpl(Graph* graph, int32_t kind, int input_cnt, int output_cnt,
               DataLayout layout)
    : graph_(reinterpret_cast<GraphImpl*>(graph)),
      kind_(kind),
      input_cnt_(input_cnt),
      output_cnt_(output_cnt),
      layout_(layout) {}

OpImpl::OpImpl(Graph* graph, DataLayout layout)
      : graph_(reinterpret_cast<GraphImpl*>(graph)),
      layout_(layout) {}

void OpImpl::SetRoundingPolicy(OverflowPolicy overflow_policy,
                               RoundingPolicy rounding_policy,
                               RoundType down_scale_size_roundin,
                               uint32_t accumulator_bits) {
  (void)overflow_policy;
  (void)rounding_policy;
  (void)down_scale_size_roundin;
  (void)accumulator_bits;
}
}  // namespace vx
}  // namespace tim
