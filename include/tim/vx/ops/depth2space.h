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
#ifndef TIM_VX_OPS_DEPTH2SPACE_H_
#define TIM_VX_OPS_DEPTH2SPACE_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## DepthToSpace
 *
 * DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
 * This is the reverse transformation of SpaceToDepth.
 *
 * Chunks of data of size block_size * block_size from depth are rearranged into
 * non-overlapping blocks of size block_size x block_size.
 *
 * The width of the output tensor is input_depth * block_size, whereas the height
 * is input_height * block_size. The depth of the input tensor must be divisible
 * by block_size * block_size
 *
 * - crop : corp the output tensor for ROI usage.
 */

class DepthToSpace : public BuiltinOp {
 public:
  enum depth2space_mode {
    DCR_mode = 0,
    CRD_mode = 1,
  };
  DepthToSpace(Graph* Graph, int block_size,
               DataLayout layout = DataLayout::WHCN);
  DepthToSpace(Graph* Graph, int block_size, depth2space_mode mode,
               DataLayout layout = DataLayout::WHCN);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  int block_size_;
  depth2space_mode mode_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_DEPTH2SPACE_H_ */
