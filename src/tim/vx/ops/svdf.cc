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
#include "tim/vx/ops/svdf.h"

#include "builtin_op_impl.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

Svdf::Svdf(Graph* graph, int32_t rank, int32_t num_units, int32_t spectrogram_length)
    : BuiltinOp(graph, VSI_NN_OP_SVDF) {
  this->impl()->node()->nn_param.svdf.rank = rank;
  this->impl()->node()->nn_param.svdf.num_units = num_units;
  this->impl()->node()->nn_param.svdf.spectrogram_length = spectrogram_length;
}

std::shared_ptr<Operation> Svdf::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<Svdf>(this->impl()->node()->nn_param.svdf.rank,
                                        this->impl()->node()->nn_param.svdf.num_units,
                                        this->impl()->node()->nn_param.svdf.spectrogram_length);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim