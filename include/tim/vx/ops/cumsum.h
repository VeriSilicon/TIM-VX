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
#ifndef TIM_VX_OPS_CUMSUM_H_
#define TIM_VX_OPS_CUMSUM_H_

#include "tim/vx/builtin_op.h"

#ifdef VSI_FEAT_OP_CUMSUM

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Cumsum
 *
 * Compute the cumulative sum of the tensor along the giveb axis. By default, it 
 * will do the sum inclusively meaning the first element is copied as is. Through 
 * an exclusive attribute, this behavior can change to exclude the first element. 
 * It can also perform summation in the opposite direction of the axis by setting 
 * reverse atrribution to 1.
 * All the attributes can be combined.
 * - axis : Specify the cumsum eperforming along which axis.Default = 0.
 * - exclusive : If exclusive = 1, perform exclusive cumsum.
 * - reverse : If reverse = 1, the cumsum is performed in the opposite direction.
 */

class CumSum : public BuiltinOp {
 public:
  CumSum(Graph* Graph, int32_t axis=0, int32_t exclusive=0, int32_t reverse=0);
  
  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;
  
 protected:
  int32_t axis_, exclusive_, reverse_;
  
 private:
  void OnBindInputPostProc(const std::shared_ptr<Tensor>& tensor, int32_t input_idx) override;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim
#endif //(VSI_FEAT_OP_CUMSUM)
#endif /* TIM_VX_OPS_CUMSUM_H_ */
