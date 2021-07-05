/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/matmul.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"
#include "type_utils.h"

namespace tim {
namespace vx {
namespace ops {

Matmul::Matmul(Graph* graph, bool transpose_a, bool transpose_b,
    bool adjoint_a, bool adjoint_b)
    : Operation(graph, VSI_NN_OP_MATRIXMUL), transpose_a_(transpose_a),
    transpose_b_(transpose_b), adjoint_a_(adjoint_a), adjoint_b_(adjoint_b) {
  this->impl()->node()->nn_param.matrixmul.transpose[0] = ToVxBool(transpose_a_);
  this->impl()->node()->nn_param.matrixmul.transpose[1] = ToVxBool(transpose_b_);
  this->impl()->node()->nn_param.matrixmul.adjoint[0] = ToVxBool(adjoint_a_);
  this->impl()->node()->nn_param.matrixmul.adjoint[1] = ToVxBool(adjoint_b_);
}

std::shared_ptr<Operation> Matmul::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Matmul>(this->transpose_a_, this->transpose_b_,
                                        this->adjoint_a_, this->adjoint_b_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim