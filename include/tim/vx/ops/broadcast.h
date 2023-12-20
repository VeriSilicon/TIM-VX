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
#ifndef OVXLIBXX_OPERATIONS_BROADCAST_H_
#define OVXLIBXX_OPERATIONS_BROADCAST_H_
#include "tim/vx/builtin_op.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Broadcast
 *
 * Broadcast an array for a compatible shape. See also numpy.broadcast_to().
 *
 * Input:
 * - input.
 *
 * Attribute:
 * - shape: the shape which broadcast to.
 * - dimensions(optional): Which dimension in the target shape each dimension
 *   of the operand shape corresponds to. For BroadcastInDim.
 */

class Broadcast : public BuiltinOp {
  public:
    Broadcast(Graph* graph, const std::vector<int32_t>& shape, const std::vector<int32_t>& dimensions = {});

    std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

   protected:
    const std::vector<int32_t> shape_;
    std::vector<int32_t> dimensions_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* OVXLIBXX_OPERATIONS_BROADCAST_H_ */
