/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
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
#include "tim/vx/ops/deconv1d.h"

#include <cassert>

#include "operation_private.h"
#include "type_utils.h"
#include "vsi_nn_pub.h"

namespace tim {
namespace vx {
namespace ops {

DeConv1d::DeConv1d(Graph* graph, int32_t oc_count, PadType pad_type,
    uint32_t ksize, uint32_t stride, uint32_t output_padding)
  : DeConv1d(graph, oc_count, pad_type, ksize, stride, output_padding,
      {0, 0}) {
}

DeConv1d::DeConv1d(Graph* graph, int32_t oc_count, PadType pad_type,
    uint32_t ksize, uint32_t stride, uint32_t output_padding,
    const std::array<uint32_t, 2>& pad, uint32_t group)
  : Operation(graph, VSI_NN_OP_DECONVOLUTION1D),
    oc_count_(oc_count),
    pad_type_(pad_type),
    ksize_(ksize),
    stride_(stride),
    output_padding_(output_padding),
    pad_(pad),
    group_(group) {

  this->impl()->node()->nn_param.deconvolution1d.ksize = ksize_;
  this->impl()->node()->nn_param.deconvolution1d.stride = stride_;
  this->impl()->node()->nn_param.deconvolution1d.pad_type = TranslatePadType(pad_type_);
  this->impl()->node()->nn_param.deconvolution1d.weights = oc_count_;
  this->impl()->node()->nn_param.deconvolution1d.group = group_;
  this->impl()->node()->nn_param.deconvolution1d.output_padding = output_padding_;
  this->impl()->node()->nn_param.deconvolution1d.pad[0] = pad_[0];
  this->impl()->node()->nn_param.deconvolution1d.pad[1] = pad_[1];
}

} // namespace ops
} // namespace vx
} // namespace tim
