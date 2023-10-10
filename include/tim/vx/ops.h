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
#ifndef TIM_VX_OPS_H_
#define TIM_VX_OPS_H_

#include "tim/vx/ops/activations.h"
#include "tim/vx/ops/addn.h"
#include "tim/vx/ops/arg.h"
#include "tim/vx/ops/batch2space.h"
#include "tim/vx/ops/batchnorm.h"
#include "tim/vx/ops/bidirectional_sequence_rnn.h"
#include "tim/vx/ops/bidirectional_sequence_rnn_ext.h"
#include "tim/vx/ops/broadcast.h"
#include "tim/vx/ops/clip.h"
#include "tim/vx/ops/concat.h"
#include "tim/vx/ops/conv1d.h"
#include "tim/vx/ops/conv2d.h"
#include "tim/vx/ops/deconv1d.h"
#include "tim/vx/ops/deconv.h"
#include "tim/vx/ops/depth2space.h"
#include "tim/vx/ops/dropout.h"
#include "tim/vx/ops/elementwise.h"
#include "tim/vx/ops/erf.h"
#include "tim/vx/ops/fullyconnected.h"
#include "tim/vx/ops/gather.h"
#include "tim/vx/ops/gather_elements.h"
#include "tim/vx/ops/gathernd.h"
#include "tim/vx/ops/groupedconv2d.h"
#include "tim/vx/ops/instancenormalization.h"
#include "tim/vx/ops/l2normalization.h"
#include "tim/vx/ops/layernormalization.h"
#include "tim/vx/ops/localresponsenormalization.h"
#include "tim/vx/ops/logical.h"
#include "tim/vx/ops/logsoftmax.h"
#include "tim/vx/ops/matmul.h"
#include "tim/vx/ops/maxpoolwithargmax.h"
#include "tim/vx/ops/maxpoolwithargmax2.h"
#include "tim/vx/ops/maxpoolgrad.h"
#include "tim/vx/ops/maxunpool2d.h"
#include "tim/vx/ops/moments.h"
#include "tim/vx/ops/nbg.h"
#include "tim/vx/ops/onehot.h"
#include "tim/vx/ops/pad.h"
#include "tim/vx/ops/pad_v2.h"
#include "tim/vx/ops/pool1d.h"
#include "tim/vx/ops/pool2d.h"
#include "tim/vx/ops/reduce.h"
#include "tim/vx/ops/relational_operations.h"
#include "tim/vx/ops/reorg.h"
#include "tim/vx/ops/reshape.h"
#include "tim/vx/ops/resize1d.h"
#include "tim/vx/ops/resize.h"
#include "tim/vx/ops/reverse.h"
#include "tim/vx/ops/rnn_cell.h"
#include "tim/vx/ops/roi_align.h"
#include "tim/vx/ops/roi_pool.h"
#include "tim/vx/ops/scatternd.h"
#include "tim/vx/ops/select.h"
#include "tim/vx/ops/shuffle_channel.h"
#include "tim/vx/ops/simple_operations.h"
#include "tim/vx/ops/signal_frame.h"
#include "tim/vx/ops/slice.h"
#include "tim/vx/ops/softmax.h"
#include "tim/vx/ops/space2batch.h"
#include "tim/vx/ops/space2depth.h"
#include "tim/vx/ops/spatial_transformer.h"
#include "tim/vx/ops/split.h"
#include "tim/vx/ops/squeeze.h"
#include "tim/vx/ops/stack.h"
#include "tim/vx/ops/stridedslice.h"
#include "tim/vx/ops/svdf.h"
#include "tim/vx/ops/tile.h"
#include "tim/vx/ops/transpose.h"
#include "tim/vx/ops/unidirectional_sequence_lstm.h"
#include "tim/vx/ops/unidirectional_sequence_rnn.h"
#include "tim/vx/ops/unidirectional_sequence_rnn_ext.h"
#include "tim/vx/ops/unstack.h"
#include "tim/vx/ops/conv3d.h"
#include "tim/vx/ops/custom_base.h"
#include "tim/vx/ops/topk.h"
#include "tim/vx/ops/tiny_yolov4_postprocess.h"
#include "tim/vx/ops/bidirectional_sequence_lstm.h"
#include "tim/vx/ops/hashtable_lookup.h"
#include "tim/vx/ops/embedding_lookup.h"
#include "tim/vx/ops/cumsum.h"
#include "tim/vx/ops/mod.h"
#include "tim/vx/ops/max_pool3d.h"
#include "tim/vx/ops/unidirectional_sequence_gru.h"
#include "tim/vx/ops/grucell.h"
#include "tim/vx/ops/scatternd_onnx_v16.h"

#endif /* TIM_VX_OPS_H_ */
