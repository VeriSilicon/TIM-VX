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
#ifndef TIM_VX_OPS_GRUCELL_H_
#define TIM_VX_OPS_GRUCELL_H_

#include <array>
#include "tim/vx/builtin_op.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## GRUCell
 *
 * - num_units : dimensionality of the output space.
 * - activation : Activation function to use.
 * - recurrent_activation : Activation function to use for the recurrent step.
 * - reset_after : whether to apply reset gate after or before matrix multiplication.
 *   False = "before", True = "after".
 */

class GRUCell : public BuiltinOp {
 public:
  enum ActivationType {
    kNONE = 0,
    kRELU = 1,
    kRELU6 = 3,
    kTANH = 4,
    kSIGMOID = 6,
    kHARDSIGMOID = 31, /* temporary use 31 */
  };

  GRUCell(Graph* graph, uint32_t num_units,
          ActivationType activation = ActivationType::kTANH,
          ActivationType recurrent_activation = ActivationType::kSIGMOID,
          vsi_bool reset_after = TRUE);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;

 protected:
  const uint32_t num_units_;
  const ActivationType activation_;
  const ActivationType recurrent_activation_;
  const int32_t reset_after_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_GRUCELL_H_ */