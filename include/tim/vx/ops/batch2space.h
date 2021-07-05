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
#ifndef TIM_VX_OPS_BATCH2SPACE_H_
#define TIM_VX_OPS_BATCH2SPACE_H_

#include <vector>

#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Batch2Space
 *
 * This operation reshapes the batch dimension (dimension 0) into M + 1 dimensions
 * of shape **block_size** + [batch], interleaves these blocks back into the grid
 * defined by the spatial dimensions [1, ..., M], to obtain a result with the same
 * rank as the input. This is the reverse transformation of Space2Batch.
 *
 * - crop : corp the output tensor for ROI usage.
 */

class Batch2Space : public Operation {
 public:
  Batch2Space(Graph* graph, const std::vector<int>& block_size,
               const std::vector<int>& crop,
               DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  std::vector<int> block_size_;
  std::vector<int> crop_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif