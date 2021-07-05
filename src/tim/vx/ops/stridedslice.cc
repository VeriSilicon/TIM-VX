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
#include "tim/vx/ops/stridedslice.h"

#include "operation_private.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

StridedSlice::StridedSlice(Graph* graph, const std::vector<int32_t> begin_dims,
                           const std::vector<int32_t> end_dims,
                           const std::vector<int32_t> stride_dims,
                           int32_t begin_mask, int32_t end_mask,
                           int32_t shrink_axis_mask)
    : Operation(graph, VSI_NN_OP_STRIDED_SLICE),
      begin_dims_(std::move(begin_dims)),
      end_dims_(std::move(end_dims)),
      stride_dims_(std::move(stride_dims)),
      begin_mask_(begin_mask),
      end_mask_(end_mask),
      shrink_axis_mask_(shrink_axis_mask) {
  this->impl()->node()->nn_param.strided_slice.begin_mask = begin_mask_;
  this->impl()->node()->nn_param.strided_slice.end_mask = end_mask_;
  this->impl()->node()->nn_param.strided_slice.shrink_axis_mask =
      shrink_axis_mask_;
  this->impl()->node()->nn_param.strided_slice.begin_dims = begin_dims_.data();
  this->impl()->node()->nn_param.strided_slice.begin_dims_num =
      begin_dims_.size();
  this->impl()->node()->nn_param.strided_slice.end_dims = end_dims_.data();
  this->impl()->node()->nn_param.strided_slice.end_dims_num = end_dims_.size();
  this->impl()->node()->nn_param.strided_slice.stride_dims =
      stride_dims_.data();
  this->impl()->node()->nn_param.strided_slice.stride_dims_num =
      stride_dims_.size();
}

std::shared_ptr<Operation> StridedSlice::Clone(
    std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<StridedSlice>(
      this->begin_dims_, this->end_dims_, this->stride_dims_, this->begin_mask_,
      this->end_mask_, this->shrink_axis_mask_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim