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
#ifndef TIM_VX_OPS_STRIDEDSLICE_H_
#define TIM_VX_OPS_STRIDEDSLICE_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## StridedSlice
 *
 * Extracts a strided slice of a tensor.
 *
 * Roughly speaking, this op extracts a slice of size (end - begin) / stride from
 * the given input tensor. Starting at the location specified by begin the slice
 * continues by adding stride to the index until all dimensions are not less than end.
 * Note that a stride can be negative, which causes a reverse slice.
 *
 * - begin_dims : the starts of the dimensions of the input tensor to be sliced.
 * - end_dims : the ends of the dimensions of the input tensor to be sliced.
 * - stride_dims : the strides of the dimensions of the input tensor to be sliced.
 * - begin_mask :  if the ith bit of begin_mask is set, begin[i] is ignored and
 * the fullest possible range in that dimension is used instead.
 * - end_mask : if the ith bit of end_mask is set, end[i] is ignored and the fullest
 * possible range in that dimension is used instead.
 * - shrink_axis_mask : if the ith bit of shrink_axis_mask is set, the ith dimension
 * specification shrinks the dimensionality by 1, taking on the value at index begin[i].
 * In this case, the ith specification must define a slice of size 1,
 * e.g. begin[i] = x, end[i] = x + 1.
 */

class StridedSlice : public Operation {
 public:
  StridedSlice(Graph* graph, const std::vector<int32_t> begin_dims,
               const std::vector<int32_t> end_dims,
               const std::vector<int32_t> stride_dims, int32_t begin_mask,
               int32_t end_mask, int32_t shrink_axis_mask);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  std::vector<int32_t> begin_dims_;
  std::vector<int32_t> end_dims_;
  std::vector<int32_t> stride_dims_;
  int32_t begin_mask_;
  int32_t end_mask_;
  int32_t shrink_axis_mask_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_STRIDEDSLICE_H_ */