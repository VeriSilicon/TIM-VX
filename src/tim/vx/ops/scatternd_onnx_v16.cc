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
#include "tim/vx/ops/scatternd_onnx_v16.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {
vsi_nn_reduction_type_e downcast_reduction_type (ScatterND_ONNX_V16::ReductionType type) {
  switch (type)
  {
    case ScatterND_ONNX_V16::ReductionType::REDUCTION_NONE:
      return VSI_NN_REDUCTION_TYPE_NONE;
    case ScatterND_ONNX_V16::ReductionType::REDUCTION_ADD:
      return VSI_NN_REDUCTION_TYPE_ADD;
    case ScatterND_ONNX_V16::ReductionType::REDUCTION_MUL:
      return VSI_NN_REDUCTION_TYPE_MUL;
    case ScatterND_ONNX_V16::ReductionType::REDUCTION_MAX:
      return VSI_NN_REDUCTION_TYPE_MAX;
    case ScatterND_ONNX_V16::ReductionType::REDUCTION_MIN:
      return VSI_NN_REDUCTION_TYPE_MIN;
    default:
      return VSI_NN_REDUCTION_TYPE_NONE;
  }
}

ScatterND_ONNX_V16::ScatterND_ONNX_V16(Graph* graph, ReductionType reduction)
    : BuiltinOp(graph, VSI_NN_OP_SCATTER_ND_UPDATE),
      reduction_(reduction) {
  this->impl()->node()->nn_param.scatter_nd_update.reduction = downcast_reduction_type(reduction_);

}

std::shared_ptr<Operation> ScatterND_ONNX_V16::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<ScatterND_ONNX_V16>(this->reduction_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim