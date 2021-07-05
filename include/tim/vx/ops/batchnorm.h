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
#ifndef OVXLIBXX_OPERATIONS_BATCHNORM_H_
#define OVXLIBXX_OPERATIONS_BATCHNORM_H_
#include "tim/vx/operation.h"

namespace tim {
namespace vx {
namespace ops {

/**
 * ## BatchNorm
 *
 * Carries out batch normalization as described in the paper
 * https://arxiv.org/abs/1502.03167.
 *
 * $$\hat x_i\leftarrow \frac{x_i-\mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$$
 *
 * $$y_i=\gamma\hat x_i+\beta\equiv BN_{\gamma,\beta}(x_i)$$
 */

class BatchNorm : public Operation {
  public:
    BatchNorm(Graph* graph, float eps);

    std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

   protected:
    float eps_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim

#endif /* OVXLIBXX_OPERATIONS_BATCHNORM_H_ */
