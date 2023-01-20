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
#ifndef TIM_VX_OPS_MOD_H_
#define TIM_VX_OPS_MOD_H_

#include "tim/vx/builtin_op.h"

#ifdef VSI_FEAT_OP_MOD

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Mod
 *
 * Mod performs element-wise binary modulus. 
 * The sign of the remainder is the same as that of the Divisor as default.
 *
 * Mod operator can also behave like C fmod() or numpy.fmod when input type is floating 
 * point. The sign of the remainder however, will be the same as the Dividend. Attribute
 * fmod is set to decide the mod behivior.
 *
 * - fmod : If the input type is floating point, then fmod must be set to 1.Default = 0
 * means integer mod.
 */

class Mod : public BuiltinOp {
 public:
  Mod(Graph* Graph, int32_t fmod = 0);
  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;
  
 protected:
  int32_t fmod_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_MOD) 
#endif /* TIM_VX_OPS_MOD_H_ */
