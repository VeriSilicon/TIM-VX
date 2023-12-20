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
#include "tim/vx/ops/cumsum.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

#ifdef VSI_FEAT_OP_CUMSUM

namespace tim {
namespace vx {
namespace ops {

CumSum::CumSum(Graph* graph, int32_t axis, int32_t exclusive, int32_t reverse)
    : BuiltinOp(graph, VSI_NN_OP_CUMSUM),  axis_(axis), exclusive_(exclusive), reverse_(reverse){
  this->impl()->node()->nn_param.cumsum.axis = axis_;
  this->impl()->node()->nn_param.cumsum.exclusive = exclusive_;
  this->impl()->node()->nn_param.cumsum.reverse = reverse_;
}

void CumSum::OnBindInputPostProc(const std::shared_ptr<Tensor>& tensor, int32_t input_idx){
  if (axis_ < 0){
    axis_ += tensor->GetShape().size();
    (void) input_idx;
    this->impl()->node()->nn_param.cumsum.axis = axis_;
  }
}

std::shared_ptr<Operation> CumSum::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<CumSum>(this->axis_, this->exclusive_, this->reverse_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_CUMSUM)
