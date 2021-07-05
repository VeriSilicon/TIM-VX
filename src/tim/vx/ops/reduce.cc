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
#include "tim/vx/ops/reduce.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

#define DEFINE_REDUCE_OP(NAME, VSI_OP_CODE)                                  \
  Reduce##NAME::Reduce##NAME(Graph* graph, const std::vector<int32_t>& axis, \
                             bool keep_dims)                                 \
      : Operation(graph, VSI_NN_OP_REDUCE),                                  \
        axis_(std::move(axis)),                                              \
        keep_dims_(keep_dims) {                                              \
    this->impl()->node()->nn_param.reduce.type = VSI_OP_CODE;                \
    this->impl()->node()->nn_param.reduce.axis = axis_.data();               \
    this->impl()->node()->nn_param.reduce.axis_num = axis_.size();           \
    this->impl()->node()->nn_param.reduce.keep_dim = keep_dims_;             \
  }                                                                          \
  std::shared_ptr<Operation> Reduce##NAME::Clone(                            \
      std::shared_ptr<Graph>& graph) const {                                 \
    return graph->CreateOperation<Reduce##NAME>(this->axis_,                 \
                                                this->keep_dims_);           \
  }

DEFINE_REDUCE_OP(Min, VSI_NN_REDUCE_MIN);
DEFINE_REDUCE_OP(Max, VSI_NN_REDUCE_MAX);
DEFINE_REDUCE_OP(Any, VSI_NN_REDUCE_ANY);
DEFINE_REDUCE_OP(All, VSI_NN_REDUCE_ALL);
DEFINE_REDUCE_OP(Prod, VSI_NN_REDUCE_PROD);
DEFINE_REDUCE_OP(Mean, VSI_NN_REDUCE_MEAN);
DEFINE_REDUCE_OP(Sum, VSI_NN_REDUCE_SUM);

#undef DEFINE_REDUCE_OP

}  // namespace ops
}  // namespace vx
}  // namespace tim
