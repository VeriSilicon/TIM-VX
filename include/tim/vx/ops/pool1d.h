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
#ifndef TIM_VX_OPS_POOL1D_H_
#define TIM_VX_OPS_POOL1D_H_

#include <array>

#include "tim/vx/builtin_op.h"
#include "tim/vx/types.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## Pool1d
 *
 * ### Classic Pool1d
 *
 * Performs an 1-D pooling operation.
 *
 * - type : MAX, AVG, L2 or AVG_ANDROID.
 * - padding : AUTO, VALID or SAME.
 * - pad : Specify the number of pad values for left, right.
 * - ksize : filter size.
 * - stride : stride along each spatial axis.
 * - round_type : CEILING or FLOOR.
 *
 * ### Global Pool1d
 *
 * - type : MAX, AVG, L2 or AVG_ANDROID.
 * - input_size : input size(only [W])
 * - round_type : CEILING or FLOOR.
 *
 * ### Adaptive Pool1d
 *
 * Same as torch.nn.AdaptiveXXXPool1d.
 *
 * - type : MAX, AVG, L2 or AVG_ANDROID.
 * - input_size : input size(only [W])
 * - output_size : output size(only [W])
 * - round_type : CEILING or FLOOR.
 *
 */

class Pool1d : public BuiltinOp {
 public:
  /* for Classic Pool1d, pool does not support auto-completion of pad value,
  you need to specify pad size explicitly, it is recommended to use the second api.*/
  Pool1d(Graph* graph, PoolType type, PadType padding,
         uint32_t ksize,
         uint32_t stride,
         RoundType round_type = RoundType::FLOOR,
         DataLayout layout = DataLayout::WCN);
  Pool1d(Graph* graph, PoolType type, const std::array<uint32_t, 2>& pad,
         uint32_t ksize,
         uint32_t stride,
         RoundType round_type = RoundType::FLOOR,
         DataLayout layout = DataLayout::WCN);

  // for Global Pool1d
  Pool1d(Graph* graph, PoolType type, uint32_t input_size,
         RoundType round_type = RoundType::FLOOR,
         DataLayout layout = DataLayout::WCN);

  // for Adaptive Pool1d
  Pool1d(Graph* graph, PoolType type, uint32_t input_size,
         uint32_t output_size,
         RoundType round_type = RoundType::FLOOR,
         DataLayout layout = DataLayout::WCN);

  std::shared_ptr<Operation> Clone(
      std::shared_ptr<Graph>& graph) const override;
  void Init();

 protected:
  const PoolType type_;
  const PadType padding_;
  uint32_t ksize_;
  uint32_t stride_;
  const RoundType round_type_;
  const std::array<uint32_t, 2> pad_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* TIM_VX_OPS_POOL1D_H_ */