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
#ifndef TIM_BATCH_FUSE_OPS_OP_BATCH_FUSE_H_
#define TIM_BATCH_FUSE_OPS_OP_BATCH_FUSE_H_

#include <memory>

#include "../batch_fuse_context.h"
#include "tim/fuse/batch_fuse.h"
#include "tim/vx/types.h"

namespace tim {
namespace fuse {

class OpBatchFuse {
 public:
  OpBatchFuse(const std::shared_ptr<vx::Operation> op,
              std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context)
      : op_(op), context_(context) {}
  virtual void OnInputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) = 0;
  virtual void OnOutputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors);

  virtual bool GapForwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) = 0;

  virtual bool GapBackwardInference(
      std::vector<std::shared_ptr<vx::Tensor>>& former_tensors) = 0;

  virtual void CloneGraph(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors);

  virtual ~OpBatchFuse() = default;

 protected:
  std::pair<uint32_t, uint32_t> ClosestFactors(uint32_t batch);

  std::vector<std::vector<int32_t>> ComputeStartPoints(
      std::vector<uint32_t> input_batch_fuse_shape,
      std::vector<uint32_t> input_shape, uint32_t batch_axis,
      std::vector<uint32_t> fuse_axes);

  std::shared_ptr<vx::Tensor> InsertPad(
      std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
      std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
      std::vector<uint32_t> fuse_axes);

  std::shared_ptr<vx::Tensor> InsertMask(
      std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
      std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
      std::vector<uint32_t> fuse_axes);

  std::shared_ptr<vx::Tensor> InsertPermuteAndReshape(
      std::shared_ptr<vx::Tensor> pad_tensor,
      std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
      std::vector<uint32_t> fuse_axes);

  std::shared_ptr<vx::Tensor> InsertSliceAndConcat(
      std::shared_ptr<vx::Tensor> input_batch_fuse_tensor,
      std::shared_ptr<vx::Tensor> input_tensor, uint32_t batch_axis,
      std::vector<uint32_t> fuse_axes);

  std::vector<std::shared_ptr<vx::Tensor>> CreateOutputsTensor();

  vx::PadType TranslatePadType(int32_t pad);
  vx::PoolType TranslatePoolType(int32_t pool);
  vx::RoundType TranslateRoundType(int32_t round);

 protected:
  const std::shared_ptr<vx::Operation> op_;
  std::shared_ptr<batch_fuse_impl::BatchFuseContext>& context_;
};

}  // namespace fuse
}  // namespace tim

#endif
