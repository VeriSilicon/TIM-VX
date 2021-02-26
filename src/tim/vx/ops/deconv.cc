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

#include "tim/vx/ops/deconv.h"

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

DeConv2d::DeConv2d(Graph* graph, int32_t weights, PadType pad_type,
    const std::array<uint32_t, 2>& ksize,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& output_padding)
  : DeConv2d(graph, weights, pad_type, ksize, stride, output_padding,
      {0, 0, 0, 0}) {
}

DeConv2d::DeConv2d(Graph* graph, int32_t weights, PadType pad_type,
    const std::array<uint32_t, 2>& ksize,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& output_padding,
    const std::array<uint32_t, 4>& pad)
  : Operation(graph, VSI_NN_OP_DECONVOLUTION),
    weights_(weights),
    pad_type_(pad_type),
    ksize_(ksize),
    stride_(stride),
    output_padding_(output_padding),
    pad_(pad) {
  this->impl()->node()->nn_param.deconv.ksize[0] = ksize_[0];
  this->impl()->node()->nn_param.deconv.ksize[1] = ksize_[1];
  this->impl()->node()->nn_param.deconv.stride[0] = stride_[0];
  this->impl()->node()->nn_param.deconv.stride[1] = stride_[1];
  this->impl()->node()->nn_param.deconv.pad_type = TranslatePadType(pad_type_);
  this->impl()->node()->nn_param.deconv.weights = weights_;
  this->impl()->node()->nn_param.deconv.group = 1;
  this->impl()->node()->nn_param.deconv.output_padding[0] = output_padding_[0];
  this->impl()->node()->nn_param.deconv.output_padding[1] = output_padding_[1];
  this->impl()->node()->nn_param.deconv.pad[0] = pad_[0];
  this->impl()->node()->nn_param.deconv.pad[1] = pad_[1];
  this->impl()->node()->nn_param.deconv.pad[2] = pad_[2];
  this->impl()->node()->nn_param.deconv.pad[3] = pad_[3];
}

} // namespace ops
} // namespace vx
} // namespace tim
