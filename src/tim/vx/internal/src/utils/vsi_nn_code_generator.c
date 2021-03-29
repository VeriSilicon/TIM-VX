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
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "vsi_nn_prv.h"
#include "vsi_nn_assert.h"
#include "vsi_nn_log.h"
#include "utils/vsi_nn_code_generator.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "utils/vsi_nn_util.h"

/* The static data file handle. */
static FILE * s_dfile_hndl = NULL;
static FILE * s_net_file_hndl = NULL;

static void _try_open_file
    (
    const char * file_path,
    FILE ** fp,
    const char * mode
    )
{
    if( NULL == file_path )
    {
        return;
    }
    if( NULL != *fp )
    {
        VSILOGW( "File handle is not NULL." );
        fclose( *fp );
    }
    *fp = fopen( file_path, mode );
    if( NULL == *fp )
    {
        VSILOGE( "Open file %s fail.", file_path );
        return;
    }
} /* _try_open_file() */

static void _try_close_file
    (
    FILE ** fp
    )
{
    if( NULL != *fp )
    {
        fflush( *fp );
        fclose( *fp );
        *fp = NULL;
    }
} /* _try_close_file() */

static void _try_pack_tensor_data
    (
    vsi_nn_graph_t *  graph,
    vsi_nn_tensor_t * tensor,
    uint64_t *       p_ofst,
    uint64_t *       p_sz
    )
{
    long ofst;
    size_t cnt;
    uint32_t  bytes;
    uint8_t * data;

    if( NULL == s_dfile_hndl || NULL == tensor
        || NULL == p_ofst || NULL == p_sz )
    {
        return;
    }
    *p_ofst = 0;
    *p_sz = 0;
    ofst = ftell( s_dfile_hndl );
    if( 0 > ofst )
    {
        VSILOGE( "Get offset error %ld.", ofst );
    }
    else
    {
        *p_ofst = (uint64_t)ofst;
        data = vsi_nn_ConvertTensorToData( graph, tensor );
        bytes = vsi_nn_GetTensorSize( tensor->attr.size,
            tensor->attr.dim_num, tensor->attr.dtype.vx_type );
        if( NULL != data )
        {
            cnt = fwrite( data, (size_t)bytes, 1, s_dfile_hndl );
            if( cnt != 1 )
            {
                VSILOGW( "Write tensor bytes(%zu/%d)", cnt, 1 );
            }
            if( cnt > 0 )
            {
                *p_sz = (uint64_t)bytes;
            }
            vsi_nn_safe_free( data );
        }
    }
} /* _pack_tensor_data() */

#define _write_code(str, ...)     _write_code_ex(str"\n", ##__VA_ARGS__)
static void _write_code_ex
    (
    const char * fmt,
    ...
    )
{
#define _MAX_LINE_SIZE      (256 - 1)

    char    line[_MAX_LINE_SIZE] = { 0 };
    int     bytes;
    va_list args;

    va_start( args, fmt );
    bytes = vsnprintf( line, _MAX_LINE_SIZE, fmt, args );
    va_end( args );

    if( NULL != s_net_file_hndl )
    {
        fwrite( line, bytes, 1, s_net_file_hndl );
    }
    else
    {
        vprintf( fmt, args );
    }
} /* _write_code() */

static void _vx_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->vx_param.has_relu = %d;",
        node_id, (int)node->vx_param.has_relu);
    _write_code("node[%u]->vx_param.overflow_policy = %#x;",
        node_id, (int)node->vx_param.overflow_policy);
    _write_code("node[%u]->vx_param.rounding_policy = %#x;",
        node_id, (int)node->vx_param.rounding_policy);
    _write_code("node[%u]->vx_param.down_scale_size_rounding = %#x;",
        node_id, (int)node->vx_param.down_scale_size_rounding);
} /* _vx_param() */

static void _conv_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->nn_param.conv2d.ksize[0] = %u;",
        node_id, node->nn_param.conv2d.ksize[0]);
    _write_code("node[%u]->nn_param.conv2d.ksize[1] = %u;",
        node_id, node->nn_param.conv2d.ksize[1]);
    _write_code("node[%u]->nn_param.conv2d.pad[0] = %u;",
        node_id, node->nn_param.conv2d.pad[0]);
    _write_code("node[%u]->nn_param.conv2d.pad[1] = %u;",
        node_id, node->nn_param.conv2d.pad[1]);
    _write_code("node[%u]->nn_param.conv2d.pad[2] = %u;",
        node_id, node->nn_param.conv2d.pad[2]);
    _write_code("node[%u]->nn_param.conv2d.pad[3] = %u;",
        node_id, node->nn_param.conv2d.pad[3]);
    _write_code("node[%u]->nn_param.conv2d.pad_type = %#x;",
        node_id, node->nn_param.conv2d.pad_type);
    _write_code("node[%u]->nn_param.conv2d.stride[0] = %u;",
        node_id, node->nn_param.conv2d.stride[0]);
    _write_code("node[%u]->nn_param.conv2d.stride[1] = %u;",
        node_id, node->nn_param.conv2d.stride[1]);
} /* _conv_param() */

static void _pool_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->nn_param.pool.ksize[0] = %u;",
        node_id, node->nn_param.pool.ksize[0]);
    _write_code("node[%u]->nn_param.pool.ksize[1] = %u;",
        node_id, node->nn_param.pool.ksize[1]);
    _write_code("node[%u]->nn_param.pool.pad[0] = %u;",
        node_id, node->nn_param.pool.pad[0]);
    _write_code("node[%u]->nn_param.pool.pad[1] = %u;",
        node_id, node->nn_param.pool.pad[1]);
    _write_code("node[%u]->nn_param.pool.pad[2] = %u;",
        node_id, node->nn_param.pool.pad[2]);
    _write_code("node[%u]->nn_param.pool.pad[3] = %u;",
        node_id, node->nn_param.pool.pad[3]);
    _write_code("node[%u]->nn_param.pool.pad_type = %#x;",
        node_id, node->nn_param.pool.pad_type);
    _write_code("node[%u]->nn_param.pool.stride[0] = %u;",
        node_id, node->nn_param.pool.stride[0]);
    _write_code("node[%u]->nn_param.pool.stride[1] = %u;",
        node_id, node->nn_param.pool.stride[1]);
    _write_code("node[%u]->nn_param.pool.type = %#x;",
        node_id, node->nn_param.pool.type);
} /* _pool_param() */

static void _lrn_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->nn_param.lrn.type = %#x;",
        node_id, node->nn_param.lrn.type);
    _write_code("node[%u]->nn_param.lrn.size = %d;",
        node_id, node->nn_param.lrn.size);
    _write_code("node[%u]->nn_param.lrn.alpha = %ff;",
        node_id, node->nn_param.lrn.alpha);
    _write_code("node[%u]->nn_param.lrn.beta = %ff;",
        node_id, node->nn_param.lrn.beta);
} /* _lrn_param() */

static void _fcl_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->nn_param.fcl.weights = %d;",
        node_id, node->nn_param.fcl.weights);
} /* _fcl_param() */

static void _concat_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _write_code("node[%u]->nn_param.concat.axis = %u;",
        node_id, node->nn_param.concat.axis);
} /* _concat_param() */

static void _conv_relu_pool_param
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    )
{
    _conv_param( node_id, node );
    _pool_param( node_id, node );
} /* _conv_relu_pool_param() */

typedef void (* _op_param_gen_t)
    (
    vsi_nn_node_id_t node_id,
    vsi_nn_node_t * node
    );

static _op_param_gen_t s_op_gen[] =
{
    /* ADD */                   NULL,
    /* MULTIPLY */              NULL,
    /* CONV2D */                _conv_param,
    /* CONV_RELU */             _conv_param,
    /* CONV_RELU_POOL */        _conv_relu_pool_param,
    /* FCL */                   _fcl_param,
    /* FCL_RELU */              _fcl_param,
    /* SOFTMAX */               NULL,
    /* POOL */                  _pool_param,
    /* LEAKY_RELU */            NULL,
    /* LRN */                   _lrn_param,
    /* CONCAT */                _concat_param,
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
    /* LRN2 */                  _lrn_param,
    /* RELATIONALOPS */         NULL,
    /* SYNC_HOST */             NULL,
    /* POW */                   NULL,
    /* FLOORDIV */              NULL,
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
    /* RNN*/                    NULL,
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
    /* SWISH */        NULL,
    /* DEPTHWISE_CONV1D */      NULL,
    /* GATHER_ND */             NULL,
    /* CAST */                  NULL,
    /* LINEAR */                NULL,
    /* BATCHNORM_SINGLE */      NULL,
    /* MOMENTS */               NULL,
    /* SQUEEZE */               NULL,
    /* HARD_SIGMOID */          NULL,
    /* MISH */                  NULL,
    /* EXPAND_BROADCAST */      NULL,
    /* PRE_PROCESS_YUV444 */    NULL,
    /* PRE_PROCESS_NV12 */      NULL,
    /* SCATTER_ND */            NULL,
    /* DECONVOLUTION1D */       NULL,
    /* INTERP */                NULL,
    /* RESIZE_1D */             NULL,
    /* UPSAMPLESCALE */         NULL,
};
_compiler_assert( _cnt_of_array(s_op_gen) == VSI_NN_OP_NUM, vsi_nn_code_generator_c );

void vsi_nn_GenGraphCCode
    (
    vsi_nn_graph_t * graph,
    const char *     net_path,
    const char *     data_path
    )
{
    uint32_t             i;
    uint32_t             j;
    uint64_t             sz;
    uint64_t             ofst;
    vsi_nn_node_t       * node;
    vsi_nn_node_id_t      node_id ;
    vsi_nn_node_id_t    * sorted_nodes;
    vsi_nn_tensor_t     * tensor;
    vsi_nn_tensor_id_t    tensor_id;

    if( NULL == graph )
    {
        return;
    }
    _try_open_file( net_path, &s_net_file_hndl, "w" );
    _try_open_file( data_path, &s_dfile_hndl, "wb" );
    VSILOGI( "Write graph ..." );
    _write_code( "\n#define load_data_to_tensor( tensor, ofst, size )    (0)\n" );
    _write_code( "vsi_nn_context_t ctx;" );
    _write_code( "vsi_nn_graph_t * graph;" );
    _write_code( "vsi_nn_node_t * node[%u];", graph->node_num );
    _write_code( "vsi_nn_tensor_id_t tensor[%u];", graph->tensor_num );
    _write_code( "vsi_nn_tensor_attr_t attr;");
    _write_code( "memset( &attr, 0, sizeof( attr ) );");
    _write_code( "ctx = vsi_nn_CreateContext();");
    _write_code( "graph = vsi_nn_CreateGraph( ctx, %u, %u );",
        graph->tensor_num, graph->node_num );
    /* Write tensors */
    for( i = 0; i < graph->tensor_num; i++ )
    {
        tensor_id = i;
        tensor = vsi_nn_GetTensor( graph, tensor_id );
        if( NULL == tensor )
        {
            continue;
        }
        _write_code( "attr.dim_num = %u;", tensor->attr.dim_num );
        _write_code( "attr.size[0] = %u;", tensor->attr.size[0] );
        _write_code( "attr.size[1] = %u;", tensor->attr.size[1] );
        _write_code( "attr.size[2] = %u;", tensor->attr.size[2] );
        _write_code( "attr.size[3] = %u;", tensor->attr.size[3] );
        _write_code( "attr.is_const = %d;", (int)tensor->attr.is_const );
        _write_code( "attr.vtl = %d;", (int)tensor->attr.vtl );
        _write_code( "attr.dtype.vx_type = %#x;", tensor->attr.dtype.vx_type );

        ofst = 0;
        sz = 0;
        if( TRUE == tensor->attr.is_const )
        {
            _try_pack_tensor_data( graph, tensor, &ofst, &sz );
        }
        _write_code( "tensor[%u] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);",
            tensor_id );
        if( sz > 0 )
        {
            _write_code( "load_data_to_tensor( tensor[%u], %llu, %llu );",
                tensor_id, ofst, sz );
        }
    }
    _write_code( "\n" );
    /* Write nodes */
    sorted_nodes = vsi_nn_SortGraphNode( graph );
    for( i = 0; i < graph->node_num; i++ )
    {
        if( NULL != sorted_nodes )
        {
            node_id = sorted_nodes[i];
        }
        else
        {
            node_id = i;
        }
        node = vsi_nn_GetNode( graph, node_id );
        _write_code( "node[%u] = vsi_nn_AppendNode( graph, %#x, NULL );",
            i, node->op );
        for( j = 0; j < node->input.num; j ++ )
        {
            if( VSI_NN_TENSOR_ID_NA != node->input.tensors[j] )
            {
                _write_code( "node[%u]->input.tensors[%d] = tensor[%u];",
                    i, j, node->input.tensors[j] );
            }
        }
        for( j = 0; j < node->output.num; j ++ )
        {
            if( VSI_NN_TENSOR_ID_NA != node->output.tensors[j] )
            {
                _write_code( "node[%u]->output.tensors[%d] = tensor[%u];",
                    i, j, node->output.tensors[j] );
            }
        }
        // write node params
        if( node->op < _cnt_of_array( s_op_gen ) )
        {
            if( NULL != s_op_gen[node->op] )
            {
                s_op_gen[node->op]( i, node );
            }
        }
        _vx_param( i, node );
        _write_code( "\n" );
    }

    if( NULL != sorted_nodes )
    {
        free( sorted_nodes );
    }
    _try_close_file( &s_dfile_hndl );
    _try_close_file( &s_net_file_hndl );
} /* vsi_nn_GenGraphCCode() */

