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
#ifndef TIM_LAYOUT_INFER_OPS_OP_LAYOUT_INFERENCE_H_
#define TIM_LAYOUT_INFER_OPS_OP_LAYOUT_INFERENCE_H_

#include <memory>

#include "../layout_infer_context.h"
#include "tim/transform/layout_inference.h"
#include "tim/vx/types.h"

namespace tim {
namespace transform {

constexpr std::initializer_list<uint32_t> kCWHN2WHCN = {1, 2, 0, 3};

constexpr std::initializer_list<uint32_t> KOcHWIc2OcIcHW = {0, 3, 1, 2};
constexpr std::initializer_list<uint32_t> kIcWHOc2WHIcOc = {1, 2, 0, 3};

constexpr std::initializer_list<uint32_t> kHWIcOc2OcIcHW = {3, 2, 0, 1};
constexpr std::initializer_list<uint32_t> kOcIcWH2WHIcOc = {2, 3, 1, 0};

class OpLayoutInfer {
 public:
  OpLayoutInfer(
      const std::shared_ptr<vx::Operation> op,
      std::shared_ptr<layout_inference_impl::LayoutInferContext>& context)
      : op_(op), context_(context) {}
  virtual void OnInputs(std::vector<std::shared_ptr<vx::Tensor>>& next_tensors) = 0;
  virtual void OnOutputs(
      std::vector<std::shared_ptr<vx::Tensor>>& next_tensors);

  virtual ~OpLayoutInfer() = default;

 protected:
  std::shared_ptr<vx::Tensor> InsertPermute(std::shared_ptr<vx::Tensor> input,
                                            std::shared_ptr<IPermuteVector> perm,
                                            bool is_graph_output = false,
                                            std::shared_ptr<vx::Tensor> src_out = nullptr);
  std::vector<std::shared_ptr<vx::Tensor>> CreateOutputsTensor(
      std::shared_ptr<IPermuteVector> required_pv);

  vx::PadType TranslatePadType(int32_t pad);

  vx::PoolType TranslatePoolType(int32_t pool);

  vx::RoundType TranslateRoundType(int32_t round);

  uint32_t MapAxis(const std::vector<uint32_t>& perm, uint32_t axis);

  std::shared_ptr<IPermuteVector> AlignPermuteVectorForMutilInputs();

  std::shared_ptr<IPermuteVector> AlignPermuteVectorForElementWise();

  void ReverseInputsPermuteVector();

  std::vector<uint32_t> GetExpandedShape(
      const std::vector<uint32_t>& ref_shape,
      const std::vector<uint32_t>& origin_shape);

  bool TransposeConstTensorData(const std::shared_ptr<vx::Tensor>& input,
                                const std::shared_ptr<IPermuteVector>& pv,
                                std::vector<uint8_t>& out_data);

  std::shared_ptr<vx::Tensor> PermuteConstTensor(
      const std::shared_ptr<vx::Tensor>& input,
      const std::shared_ptr<IPermuteVector>& pv);
  
  std::vector<uint32_t> MapMultipleAxis(const std::vector<uint32_t>& perm,
                                   const std::vector<uint32_t>& axises);
  std::vector<int32_t> MapMultipleAxis(const std::vector<uint32_t>& perm,
                                   const std::vector<int32_t>& axises);

 protected:
  const std::shared_ptr<vx::Operation> op_;
  std::shared_ptr<layout_inference_impl::LayoutInferContext>& context_;
};

}  // namespace transform
}  // namespace tim

#endif