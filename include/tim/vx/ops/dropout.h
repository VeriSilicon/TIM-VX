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
#ifndef OVXLIBXX_OPERATIONS_DROPOUT_H_
#define OVXLIBXX_OPERATIONS_DROPOUT_H_
#include "tim/vx/operation.h"


namespace tim {
namespace vx {
namespace ops {

/**
 * ## Dropout
 *
 * The Dropout layer randomly sets input units to 0 with a frequency of rate at
 * each step during training time, which helps prevent overfitting.
 *
 * TIM-VX only focus on inference time, and just scaling input tensor by **ratio**
 * for Dropout operator.
 */

class Dropout : public Operation {
 public:
  Dropout(Graph* graph, float ratio);

  std::shared_ptr<Operation> Clone(std::shared_ptr<Graph>& graph) const override;

 protected:
  float ratio_;
};

}  // namespace ops
}  // namespace vx
}  // namespace tim


#endif /* OVXLIBXX_OPERATIONS_DROPOUT_H_ */
