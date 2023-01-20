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
#ifndef TIM_VX_OPS_BIDIRECTIONAL_SEQUENCE_LSTM_H_
#define TIM_VX_OPS_BIDIRECTIONAL_SEQUENCE_LSTM_H_

#include "tim/vx/operation.h"
namespace tim {
namespace vx {
namespace ops {

class BidirectionalSequenceLstm : public Operation {
 public:
  enum ActivationType {
    kNONE = 0,
    kRELU = 1,
    kRELU1 = 2,
    kRELU6 = 3,
    kTANH = 4,
    kSIGMOID = 6,
    kHARDSIGMOID = 31, /* temporary use 31 */
  };
  BidirectionalSequenceLstm(
          Graph* graph, float cell_clip, float proj_clip,
          ActivationType act_type, float forget_bias, bool time_major = false,
          ActivationType recurrent_act_type = ActivationType::kSIGMOID,
          bool return_sequences = false /*False: only return last state*/
      );

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  const float cell_clip_;
  const float proj_clip_;
  const ActivationType act_type_;
  const float forget_bias_;
  const bool time_major_;
  const ActivationType recurrent_act_type_;
  const bool return_sequences_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_BIDIRECTIONAL_SEQUENCE_LSTM_H_ */