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
#ifndef TIM_VX_OPS_BIDIRECTIONAL_SEQUENCE_RNN_EXT_H_
#define TIM_VX_OPS_BIDIRECTIONAL_SEQUENCE_RNN_EXT_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {
    /**
     * ## Bidirectional sequence rnn for onnx
     *  how to bind input/output: take unidirectional_sequence_rnn_ext_test.cc
     */
    class BidirectionalSequenceRnnExt: public Operation {
     public:
      BidirectionalSequenceRnnExt(
          Graph* graph, 
          tim::vx::ops::BidirectionalSequenceRnn::ActivationType act_type
      );

      std::shared_ptr<Operation> Clone(
          std::shared_ptr<Graph>& graph) const override;

     protected:
      tim::vx::ops::BidirectionalSequenceRnn::ActivationType act_type_;
    };
}
}  // namespace vx
}  // namespace tim

#endif
