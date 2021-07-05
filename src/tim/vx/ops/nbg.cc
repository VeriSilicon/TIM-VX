/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
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
#include "tim/vx/ops/nbg.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
NBG::NBG(Graph* graph, const char* binary, size_t input_count,
         size_t output_count)
    : Operation(graph, VSI_NN_OP_NBG, input_count, output_count) {
  this->impl()->node()->nn_param.nbg.url = binary;
  this->impl()->node()->nn_param.nbg.type = VSI_NN_NBG_POINTER;
}

std::shared_ptr<Operation> NBG::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<NBG>(this->impl_->node_->nn_param.nbg.url,
                                     this->impl_->input_cnt_,
                                     this->impl_->output_cnt_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim