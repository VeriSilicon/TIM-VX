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
#ifndef TIM_VX_OPS_SPACE2BATCH_H_
#define TIM_VX_OPS_SPACE2BATCH_H_

#include <vector>

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Space2Batch
 *
 * This operation divides "spatial" dimensions [1, ..., M] of the input into a grid
 * of blocks of shape **block_size**, and interleaves these blocks with the "batch"
 * dimension (0) such that in the output, the spatial dimensions [1, ..., M] correspond
 * to the position within the grid, and the batch dimension combines both the position
 * within a spatial block and the original batch position. Prior to division into blocks,
 * the spatial dimensions of the input are optionally zero padded according to paddings.
 * This is the reverse transformation of Batch2Space.
 *
 * - pad : the paddings for each spatial dimension of the input tensor.
 */

class Space2Batch : public Operation {
 public:
  Space2Batch(Graph* graph, const std::vector<int>& block_size,
               const std::vector<int>& pad,
               DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  std::vector<int> block_size_;
  std::vector<int> pad_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_SPACE2BATCH_H_ */