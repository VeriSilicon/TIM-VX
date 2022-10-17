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
#include <string.h>

#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_types.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_assert.h"
#include "utils/vsi_nn_util.h"

typedef void ( *_node_template )( vsi_nn_node_t * );

static void _template_pool( vsi_nn_node_t* node );

static void _template_conv2d( vsi_nn_node_t* node );

static void _template_conv_relu( vsi_nn_node_t* node );

static void _template_conv_relu_pool( vsi_nn_node_t* node );

static void _template_fcl( vsi_nn_node_t* node );

static void _template_fcl_relu( vsi_nn_node_t* node );

static void _template_lrn( vsi_nn_node_t* node );

static _node_template s_template[] =
{
    /* ADD */                   NULL,
    /* MULTIPLY */              NULL,
    /* CONV2D */                _template_conv2d,
    /* CONV_RELU */             _template_conv_relu,
    /* CONV_RELU_POOL */        _template_conv_relu_pool,
    /* FCL */                   _template_fcl,
    /* FCL_RELU */              _template_fcl_relu,
    /* SOFTMAX */               NULL,
    /* POOL */                  _template_pool,
    /* LEAKY_RELU */            NULL,
    /* LRN */                   _template_lrn,
    /* CONCAT */                NULL,
    /* SPLIT */                 NULL,
    /* NOOP */                  NULL,
    /* ROI_POOL */              NULL,
    /* BATCH_NORM */            NULL,
    /* PROPOSAL */              NULL,
    /* DECONVOLUTION */         NULL,
    /* RESHAPE */               NULL,
    /* PERMUTE */               NULL,
    /* PRELU */                 NULL,
    /* UPSAMPLE */              NULL,
    /* RELU */                  NULL,
    /* RELUN */                 NULL,
    /* LSTM */                  NULL,
    /* REORG */                 NULL,
    /* VARIABLE */              NULL,
    /* L2_NORMALIZE */          NULL,
    /* FCL2 */                  NULL,
    /* POOLWITHARGMAX */        NULL,
    /* ARGMAX */                NULL,
    /* MAXIMUM */               NULL,
    /* L2NORMALIZESCALE */      NULL,
    /* CROP */                  NULL,
    /* SUBTRACT */              NULL,
    /* RELU6 */                 NULL,
    /* SIGMOID */               NULL,
    /* TANH */                  NULL,
    /* SQRT */                  NULL,
    /* RSQRT */                 NULL,
    /* SOFTRELU */              NULL,
    /* DIVIDE */                NULL,
    /* DROPOUT */               NULL,
    /* SHUFFLECHANNEL */        NULL,
    /* RESIZE */                NULL,
    /* REVERSE */               NULL,
    /* DEPTH2SPACE */           NULL,
    /* SPACE2DEPTH */           NULL,
    /* DATACONVERT */           NULL,
    /* SCALE */                 NULL,
    /* SLICE */                 NULL,
    /* ELU */                   NULL,
    /* BATCH2SPACE */           NULL,
    /* SPACE2BATCH */           NULL,
    /* PAD */                   NULL,
    /* IMAGEPROCESS */          NULL,
    /* MATRIXMUL */             NULL,
    /* LSTMUNIT */              NULL,
    /* LAYERNORM */             NULL,
    /* REDUCE */                NULL,
    /* INSTANCENORM */          NULL,
    /* TENSORSTACKCONCAT */     NULL,
    /* STRIDED_SLICE */         NULL,
    /* SIGNALFRAME */           NULL,
    /* A_TIMES_B_PLUS_C */      NULL,
    /* SVDF */                  NULL,
    /* ABS */                   NULL,
    /* CONV1D */                NULL,
    /* NBG */                   NULL,
    /* CONCATSHIFT */           NULL,
    /* LRN2 */                  _template_lrn,
    /* RELATIONALOPS */         NULL,
    /* SYNC_HOST */             NULL,
    /* POW */                   NULL,
    /* FOORDIV */               NULL,
    /* MINIMUM */               NULL,
    /* SPATIAL_TRANSFORMER */   NULL,
    /* LOGICAL_OPS */           NULL,
    /* SELECT */                NULL,
    /* LSTMUNIT_ACTIVATION */   NULL,
    /* LSTMUNIT_OVXLIB */       NULL,
    /* TENSOR_ADD_MEAN_STDDEV_NORM */ NULL,
    /* RELU1 */                 NULL,
    /* STACK */                 NULL,
    /* FLOOR */                 NULL,
    /* SQUARE */                NULL,
    /* NEG */                   NULL,
    /* EXP */                   NULL,
    /* LSTM_OVXLIB */           NULL,
    /* PRE_PROCESS_TENSOR */    NULL,
    /* HASHTABLE_LOOKUP */      NULL,
    /* EMBEDDING_LOOKUP */      NULL,
    /* LSH_PROJECTION */        NULL,
    /* RNN */                   NULL,
    /* CLIP */                  NULL,
    /* POST_PROCESS */          NULL,
    /* PRE_PROCESS_GRAY */      NULL,
    /* UNSTACK */               NULL,
    /* PRE_PROCESS_RGB */       NULL,
    /* PRE_PROCESS */           NULL,
    /* ADDN */                  NULL,
    /* PRE_PROCESS_YUV420 */    NULL,
    /* EXTRA_ENDING */          NULL,
    /* GATHER */                NULL,
    /* TILE */                  NULL,
    /* GROUPED_CONV2D */        NULL,
    /* TOPK */                  NULL,
    /* PRE_PROCESS_BGRA */      NULL,
    /* LOGICAL_NOT */           NULL,
    /* SIN */                   NULL,
    /* LOG */                   NULL,
    /* ARGMIN */                NULL,
    /* ROI_ALIGN */             NULL,
    /* HEATMAP_MAX_KEYPOINT */  NULL,
    /* AXIS_ALIGNED_BBOX_TRANSFORM */ NULL,
    /* BOX_WITH_NMS_LIMIT */    NULL,
    /* GENERATE_PROPOSALS */    NULL,
    /* DETECTION_POSTPROCESS */ NULL,
    /* RANDOM_MULTINOMIAL */    NULL,
    /* LOG_SOFTMAX */           NULL,
    /* RELU_KERAS */            NULL,
    /* GRU_OVXLIB */            NULL,
    /* GRUCELL_OVXLIB */        NULL,
    /* UNIDIRECTIONAL_SEQUENCE_RNN */ NULL,
    /* QUANTIZED_16BIT_LSTM */  NULL,
    /* BIDIRECTIONAL_SEQUENCE_RNN */ NULL,
    /* BIDIRECTIONAL_SEQUENCE_LSTM */ NULL,
    /* RNNCELL_OVXLIB */        NULL,
    /* SWISH */                 NULL,
    /* GATHER_ND */             NULL,
    /* CAST */                  NULL,
    /* LINEAR */                NULL,
    /* MOMENTS */               NULL,
    /* PRE_PROCESS_YUV444 */    NULL,
    /* PRE_PROCESS_NV12 */      NULL,
    /* SCATTER_ND */            NULL,
    /* DECONVOLUTION1D */       NULL,
    /* GROUPNORM */             NULL,
    /* SEQUENCE_MASK */         NULL,
    /* REPEAT */                NULL,
    /* SCATTER_ND_UPDATE */     NULL,
    /* CONV2D_LSTM */           NULL,
    /* CONV2D_LSTM_CELL */      NULL,
    /* GRU */                   NULL,
    /* GRUCELL */               NULL,
    /* GRUCELL_ACTIVATION */    NULL,
    /* CUMSUM */                NULL,
    /* MAXPOOLWITHARGMAX */     NULL,
    /* MOD */                   NULL,
    /* LPPOOL */                NULL,
    /* PRE_PROCESS_YUV422 */    NULL,
};
//_compiler_assert( _cnt_of_array(s_template) == VSI_NN_OP_NUM, vsi_nn_node_attr_template_c );

void vsi_nn_apply_node_attr_template
    ( vsi_nn_node_t * node )
{
    if( node->op >= _cnt_of_array( s_template ) )
    {
        VSILOGW( "Unsupport operation id %d.", node->op );
        return;
    }
    if( NULL != s_template[node->op] )
    {
        s_template[node->op]( node );
    }
} /* ovx_apply_node_attr_template() */

static void _template_lrn
    ( vsi_nn_node_t* node )
{
    node->nn_param.lrn.bias = 1.0f;
    node->nn_param.lrn.type = VX_CONVOLUTIONAL_NETWORK_NORM_ACROSS_MAPS;
    node->nn_param.lrn.size = 5;
    node->nn_param.lrn.alpha = 0.0001f;
    node->nn_param.lrn.beta = 0.75f;
} /* _template_lrn() */

static void _template_pool
    ( vsi_nn_node_t * node )
{
    node->nn_param.pool.ksize[0] = 1;
    node->nn_param.pool.ksize[1] = 1;
    node->nn_param.pool.stride[0] = 1;
    node->nn_param.pool.stride[1] = 1;
    node->nn_param.pool.pad[0] = 0;
    node->nn_param.pool.pad[1] = 0;
    node->nn_param.pool.pad[2] = 0;
    node->nn_param.pool.pad[3] = 0;
    node->nn_param.pool.pad_type = VSI_NN_PAD_AUTO;
    node->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    node->nn_param.pool.round_type = VSI_NN_ROUND_CEIL;
    node->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;
} /* _template_pool() */

static void _template_conv2d
    ( vsi_nn_node_t * node )
{
    node->nn_param.conv2d.ksize[0] = 1;
    node->nn_param.conv2d.ksize[1] = 1;
    node->nn_param.conv2d.weights = 1;
    node->nn_param.conv2d.stride[0] = 1;
    node->nn_param.conv2d.stride[1] = 1;
    node->nn_param.conv2d.pad[0] = 0;
    node->nn_param.conv2d.pad[1] = 0;
    node->nn_param.conv2d.pad[2] = 0;
    node->nn_param.conv2d.pad[3] = 0;
    node->nn_param.conv2d.pad_type = VSI_NN_PAD_AUTO;
    node->nn_param.conv2d.group = 1;
    node->vx_param.has_relu = FALSE;
} /* _template_conv2d() */

static void _template_conv_relu
    ( vsi_nn_node_t * node )
{
    _template_conv2d( node );
    node->vx_param.has_relu = TRUE;
} /* _template_conv_relu() */

static void _template_conv_relu_pool
    ( vsi_nn_node_t * node )
{
    _template_conv_relu( node );
    _template_pool( node );
} /* _template_conv_relu_pool() */

static void _template_fcl
    ( vsi_nn_node_t * node )
{
    node->nn_param.fcl.weights = 1;
    node->vx_param.has_relu = FALSE;
} /* _template_fcl() */

static void _template_fcl_relu
    ( vsi_nn_node_t* node )
{
    _template_fcl( node );
    node->vx_param.has_relu = TRUE;
} /* _template_fcl_relu() */

