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
#include "tim/vx/ops/max_pool3d.h"
#include "type_utils.h"
#include "builtin_op_impl.h"
#include "vsi_nn_pub.h"

#ifdef VSI_FEAT_OP_MAX_POOL3D

namespace tim {
namespace vx {
namespace ops {
MaxPool3d::MaxPool3d(Graph* Graph, RoundType round_type,
                     const std::array<uint32_t, 3>& ksize, 
                     const std::array<uint32_t, 3>& stride,
                     const std::array<uint32_t, 6>& pad,
                     PadType pad_type,
                     DataLayout layout)
     : BuiltinOp(Graph, VSI_NN_OP_MAX_POOL3D, 1, 1, layout), 
                 round_type_(round_type),
                 ksize_(ksize),
                 stride_(stride),
                 pad_(pad),
                 pad_type_(pad_type){
  this->impl()->node()->nn_param.max_pool3d.round_type = TranslateRoundType(round_type_);
  this->impl()->node()->nn_param.max_pool3d.ksize[0] = ksize_[0];
  this->impl()->node()->nn_param.max_pool3d.ksize[1] = ksize_[1];
  this->impl()->node()->nn_param.max_pool3d.ksize[2] = ksize_[2];
  this->impl()->node()->nn_param.max_pool3d.stride[0] = stride_[0];
  this->impl()->node()->nn_param.max_pool3d.stride[1] = stride_[1];
  this->impl()->node()->nn_param.max_pool3d.stride[2] = stride_[2];
  for (int i = 0; i < 6; i++){this->impl()->node()->nn_param.max_pool3d.pad[i] = pad_[i];}
  this->impl()->node()->nn_param.max_pool3d.pad_type = TranslatePadType(pad_type_);
}

std::shared_ptr<Operation> MaxPool3d::Clone(std::shared_ptr<Graph>& graph) const {
  return graph->CreateOperation<MaxPool3d>(
    this->round_type_, this->ksize_, this->stride_, this->pad_, this->pad_type_);
}

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif //(VSI_FEAT_OP_MAX_POOL3D)
