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
#include "tim/vx/ops/reshape.h"

#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

#include <algorithm>

namespace tim {
namespace vx {
namespace ops {
class ReshapeImpl : public BuiltinOpImpl {
  public:
   ReshapeImpl(Graph* graph, const std::vector<vsi_size_t>& shape)
       : BuiltinOpImpl(graph, 
       #ifdef _VSI_NN_OP_RESHAPE2_H
       VSI_NN_OP_RESHAPE2
       #else
       VSI_NN_OP_RESHAPE
       #endif
       , 1, 1), shape_(shape) {}

   std::vector<vsi_size_t> shape_;
};

Reshape::Reshape(Graph* graph, const std::vector<uint32_t>& size)
{
  std::vector<vsi_size_t> shape;
  std::transform(size.begin(), size.end(), std::back_inserter(shape), [](const uint32_t& d){
    return static_cast<vsi_size_t>(d);
  });

  auto lcl_impl = std::make_unique<ReshapeImpl>(graph, shape);
  
  #ifdef _VSI_NN_OP_RESHAPE2_H
  lcl_impl->node()->nn_param.reshape2.size = lcl_impl->shape_.data();
  lcl_impl->node()->nn_param.reshape2.dim_num = size.size();
  #else
  lcl_impl->node()->nn_param.reshape.size = lcl_impl->shape_.data();
  lcl_impl->node()->nn_param.reshape.dim_num = size.size();
  #endif

  impl_.reset(dynamic_cast<OpImpl*>(lcl_impl.release()));
}

std::shared_ptr<Operation> Reshape::Clone(
    std::shared_ptr<Graph>& graph) const {
  std::vector<uint32_t> size;
  const ReshapeImpl* lcl_impl = (dynamic_cast<ReshapeImpl*>(impl_.get()));
  std::transform(lcl_impl->shape_.begin(), lcl_impl->shape_.end(), std::back_inserter(size), [](const vsi_size_t& d){
    return static_cast<uint32_t>(d);
  });

  return graph->CreateOperation<Reshape>(size);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim