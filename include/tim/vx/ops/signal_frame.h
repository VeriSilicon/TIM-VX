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
#ifndef TIM_VX_OPS_SIGNALFRAME_H_
#define TIM_VX_OPS_SIGNALFRAME_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Signalframe
 *
 * ```
 *  tf.signal.frame(
        signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=0, name=None
    )     : Expands signal's axis dimension into frames of frame_length.
 * ```
 */

class SignalFrame : public BuiltinOp {
  public:
   SignalFrame(Graph* graph, uint32_t window_length, uint32_t step, uint32_t pad_end=0,
      uint32_t axis=0);

   std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

  protected:
   const uint32_t window_length_;
   const uint32_t step_;
   const uint32_t pad_end_;
   const uint32_t axis_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_SIGNALFRAME_H_ */