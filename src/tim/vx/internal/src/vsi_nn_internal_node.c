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
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_types.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_rnn.h"
#include "vsi_nn_test.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_map.h"

/**********************************************************
* MACROS
**********************************************************/
#define LINKLIST_APPEND( _HEAD, _ITEM ) {                \
    vsi_nn_LinkListPushEnd((vsi_nn_link_list_t **)&(_HEAD), \
    (vsi_nn_link_list_t *)(_ITEM) ); }

#define WKSP(_NODE_PTR) ((vsi_nn_internal_node_wksp_t *)    \
    ((_NODE_PTR)->internal_node_wksp))

/**********************************************************
* LOCAL FUNCTIONS
**********************************************************/
static vsi_nn_internal_node_t* vsi_nn_internal_create_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_op_t op,
    vsi_size_t input_num,
    vsi_size_t output_num
    )
{
    vsi_nn_internal_node_t* node = NULL;
    vsi_nn_node_t* n = NULL;
    vsi_nn_tensor_t** inputs = NULL;
    vsi_nn_tensor_t** outputs = NULL;

    node = (vsi_nn_internal_node_t *)malloc( sizeof(vsi_nn_internal_node_t) );
    if( node )
    {
        memset(node, 0x00, sizeof(vsi_nn_internal_node_t) );

        n = vsi_nn_NewNode( graph, op, input_num, output_num );
        if( n )
        {
            inputs = (vsi_nn_tensor_t **)malloc( n->input.num * sizeof(vsi_nn_tensor_t*));
            if( inputs )
            {
                memset( inputs, 0x00, ( n->input.num * sizeof(vsi_nn_tensor_t*)) );
            }
            outputs = (vsi_nn_tensor_t **)malloc( n->output.num * sizeof(vsi_nn_tensor_t*));
            if( outputs )
            {
                memset( outputs, 0x00, ( n->output.num * sizeof(vsi_nn_tensor_t*)) );
            }
        }
    }

    if( node && n && inputs && outputs )
    {
        node->node = n;
        node->inputs = inputs;
        node->outputs = outputs;

        return node;
    }
    else
    {
        if(n)
        {
            vsi_nn_ReleaseNode(&n);
            n = NULL;
        }
        if(inputs)
        {
            free(inputs);
            inputs = NULL;
        }
        if(outputs)
        {
            free(outputs);
            outputs = NULL;
        }
        vsi_nn_internal_release_node( &node );
        return NULL;
    }
} /* vsi_nn_internal_create_node() */

static vsi_nn_internal_tensor_t* vsi_nn_internal_create_tensor
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_attr_t* attr,
    float default_value
    )
{
    vsi_nn_internal_tensor_t* tensor = NULL;

    if( !graph || !attr )
    {
        return tensor;
    }

    tensor = (vsi_nn_internal_tensor_t *)malloc( sizeof(vsi_nn_internal_tensor_t) );
    if( tensor )
    {
        memset( tensor, 0x00, sizeof(vsi_nn_internal_tensor_t) );
        if( attr->is_const )
        {
            tensor->t = vsi_nn_CreateTensorWithDefault( graph, attr, default_value );
        }
        else
        {
            tensor->t = vsi_nn_CreateTensor( graph, attr );
        }

        if( !tensor->t )
        {
            vsi_nn_internal_release_tensor( &tensor );
        }
    }

    return tensor;
} /* vsi_nn_internal_create_tensor() */

/**********************************************************
* PUBLIC FUNCTIONS
**********************************************************/
vsi_nn_internal_tensor_t* vsi_nn_internal_create_zero_bias_tensor
    (
    vsi_nn_node_t* node,
    vsi_nn_tensor_attr_t* input_attr,
    vsi_nn_tensor_attr_t* weight_attr,
    vsi_nn_op_t op,
    vsi_bool use_virtual_tensor
    )
{
    vsi_nn_tensor_attr_t attr;
    float scale = 1.0f;
    int8_t fl = 0;

    memset(&attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

    /* create zero bias for NN/TP */
    switch(op)
    {
        case VSI_NN_OP_FCL:
        case VSI_NN_OP_FCL2:
        case VSI_NN_OP_FCL_RELU:
            attr.size[0] = weight_attr->size[1];
            break;
        case VSI_NN_OP_CONV2D:
        case VSI_NN_OP_CONV_RELU:
        case VSI_NN_OP_CONV_RELU_POOL:
        case VSI_NN_OP_GROUPED_CONV2D:
            attr.size[0] = weight_attr->size[3];
            break;
        default:
            attr.size[0] = weight_attr->size[1]; // default is FC
            VSILOGW("Ovxlib only auto fill bias for conv2d and fc, but current op is %s\n",
                vsi_nn_OpGetName(op));
            break;
    }
    attr.dim_num = 1;
    attr.vtl = use_virtual_tensor;
    attr.is_const = !use_virtual_tensor;

    if(input_attr->dtype.qnt_type != VSI_NN_QNT_TYPE_NONE &&
        input_attr->dtype.qnt_type != weight_attr->dtype.qnt_type)
    {
        VSILOGE("input qnt_type[%d] != weight qnt_type[%d]",
            input_attr->dtype.qnt_type, weight_attr->dtype.vx_type);
        return NULL;
    }

    if (weight_attr->dtype.qnt_type == VSI_NN_QNT_TYPE_NONE)
    {
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
    }
    else
    {
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
    }

    switch(input_attr->dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
            scale = input_attr->dtype.scale;
            break;

        case VSI_NN_QNT_TYPE_DFP:
            fl = input_attr->dtype.fl;
            break;

        case VSI_NN_QNT_TYPE_NONE:
            scale = 1.0f;
            fl = 0;
            break;

        default:
            VSILOGE("Unsupported quantization type: %d", input_attr->dtype.qnt_type);
            break;
    }

    switch(weight_attr->dtype.qnt_type)
    {
        case VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC:
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
        case VSI_NN_QNT_TYPE_SYMMETRIC_FLOAT8:
            attr.dtype.scale = weight_attr->dtype.scale * scale;
            attr.dtype.zero_point = 0;
            attr.dtype.qnt_type = weight_attr->dtype.qnt_type;
            break;

        case VSI_NN_QNT_TYPE_DFP:
            attr.dtype.fl = weight_attr->dtype.fl + fl;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_DFP;
            break;

        case VSI_NN_QNT_TYPE_NONE:
            break;

        default:
            VSILOGE("Unsupported quantization type: %d", weight_attr->dtype.qnt_type);
            break;
    }

    return vsi_nn_internal_new_tensor(node, &attr, 0.0f);
} /* vsi_nn_internal_create_zero_bias_tensor() */

vsi_status vsi_nn_internal_deinit_node
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;

    curr = WKSP(node)->nodes;
    while( NULL != curr )
    {
        VSILOGD("Optimize node uid[%u] sub_uid[%u] op[%s]",
            node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

        status = vsi_nn_OpDeinit( curr->node->op, curr->node );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("op_optimize fail %d", curr->node->op);
            break;
        }

        curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
    }

    return status;
} /* vsi_nn_internal_deinit_node() */

vsi_status vsi_nn_internal_deinit_node_wksp
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* head = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_nn_internal_tensor_t* tensor_head = NULL;
    vsi_nn_internal_tensor_t* tensor_curr = NULL;

    if( node && node->internal_node_wksp )
    {
        head = WKSP(node)->nodes;
        while( NULL != head )
        {
            curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&head );
            vsi_nn_internal_release_node( &curr );
        }

        tensor_head = WKSP(node)->tensors;
        while( NULL != tensor_head )
        {
            tensor_curr = (vsi_nn_internal_tensor_t *)vsi_nn_LinkListPopStart(
                (vsi_nn_link_list_t **)&tensor_head );
            vsi_nn_internal_release_tensor( &tensor_curr );
        }

        free( node->internal_node_wksp );
        node->internal_node_wksp = NULL;
    }

    return status;
} /* vsi_nn_internal_deinit_node_wksp() */

void vsi_nn_internal_dump_node_output
    (
    vsi_nn_graph_t* graph,
    const char* path,
    const char* filename_prefix,
    vsi_bool force_fp32,
    vsi_nn_node_t* node
    )
{
#define _MAX_TENSOR_NAME_SZ (1024)
#define _SHAPE_BUF_SZ   (64)
    char shape[_SHAPE_BUF_SZ] = { 0 };
    char filename[_MAX_TENSOR_NAME_SZ] = { 0 };
    const char* op_name;
    uint32_t o;
    vsi_nn_internal_node_t* head = ((vsi_nn_internal_node_wksp_t *)node->internal_node_wksp)->nodes;
    while( NULL != head )
    {
        vsi_nn_internal_node_t* curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListPopStart(
            (vsi_nn_link_list_t **)&head );
        if( curr )
        {
            if (curr->node->internal_node_wksp)
            {
                vsi_nn_internal_dump_node_output(graph, path, filename_prefix,
                    force_fp32, curr->node);
            }
            else
            {
                for( o = 0; o < curr->node->output.num; o++ )
                {
                    vsi_nn_tensor_t* tensor = curr->outputs[o];
                    if( tensor )
                    {
                        if( TRUE == tensor->attr.vtl )
                        {
                            VSILOGW("Uid %u node's tensor %d is virtual",
                                curr->node->uid, o);
                            continue;
                        }
                        // TODO: Support different tensor format
                        vsi_nn_ShapeToString( tensor->attr.size, tensor->attr.dim_num,
                            shape, _SHAPE_BUF_SZ, FALSE );
                        op_name = vsi_nn_OpGetName( curr->node->op );
                        snprintf( filename, _MAX_TENSOR_NAME_SZ,
                            "%s/%s%s_uid_%u_sub_%u_t_%u_s_%s.txt", path, filename_prefix,
                            op_name, node->uid, curr->node->uid, o, shape);
                        if( FALSE == force_fp32 )
                        {
                            vsi_nn_SaveTensorToText( graph, tensor, filename, NULL );
                        }
                        else
                        {
                            vsi_nn_SaveTensorToTextByFp32( graph, tensor, filename, NULL );
                        }
                    }
                }
            }
        }
    }
} /* vsi_nn_internal_dump_node_output() */

vsi_nn_internal_node_t* vsi_nn_internal_get_node_by_uid
    (
    vsi_nn_node_t* node,
    int uid
    )
{
    vsi_nn_internal_node_t* curr = NULL;

    if( node && node->internal_node_wksp )
    {
        curr = WKSP(node)->nodes;
        while( NULL != curr )
        {
            if( curr->node->uid == (uint32_t)uid )
            {
                return curr;
            }

            curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
        }
    }

    return NULL;
} /* vsi_nn_internal_get_node_by_uid() */

vsi_status vsi_nn_internal_init_node_wksp
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_internal_node_wksp_t* wksp = NULL;

    if( node->internal_node_wksp )
    {
        vsi_nn_internal_deinit_node_wksp( node );
    }

    wksp = (vsi_nn_internal_node_wksp_t *)malloc( sizeof( vsi_nn_internal_node_wksp_t ) );
    if( wksp )
    {
        memset( wksp, 0x00, sizeof( vsi_nn_internal_node_wksp_t ) );
        wksp->curr_node_uid = 1;

        node->internal_node_wksp = wksp;

        status = VSI_SUCCESS;
    }

    return status;
} /* vsi_nn_internal_init_node_wksp() */

void vsi_nn_internal_init_tensor_attr
    (
    vsi_nn_tensor_attr_t* attr,
    const vsi_nn_dtype_t* dtype,
    vsi_bool use_virtual_tensor
    )
{
    memset(attr, 0x00, sizeof(vsi_nn_tensor_attr_t));

    //memset(attr->size, 0, VSI_NN_MAX_DIM_NUM * sizeof(vsi_size_t));
    attr->dim_num = VSI_NN_DIM_AUTO;
    attr->vtl = use_virtual_tensor;
    attr->is_const = FALSE;

    if( dtype->qnt_type == VSI_NN_QNT_TYPE_NONE &&
        ( dtype->vx_type != VSI_NN_TYPE_FLOAT16 &&
          dtype->vx_type != VSI_NN_TYPE_FLOAT32 &&
          dtype->vx_type != VSI_NN_TYPE_BFLOAT16 ) )
    {
        attr->dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        attr->dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    }
    else
    {
        memcpy(&attr->dtype, dtype, sizeof(vsi_nn_dtype_t));
    }
} /* vsi_nn_internal_init_tensor_attr() */

vsi_nn_internal_node_t* vsi_nn_internal_new_node
    (
    vsi_nn_node_t* node,
    vsi_nn_op_t op,
    vsi_size_t input_num,
    vsi_size_t output_num
    )
{
    vsi_nn_internal_node_t* inode = NULL;

    inode = vsi_nn_internal_create_node( node->graph,
                op, input_num, output_num );
    if (inode)
    {
        inode->node->attr.const_tensor_preload_type = node->attr.const_tensor_preload_type;
    }
    return inode;
} /* vsi_nn_internal_new_node() */

void* vsi_nn_internal_new_node_param
    (
    vsi_nn_internal_node_t* inode,
    size_t size /* in bytes */
    )
{
    vsi_nn_internal_node_param_t* param = NULL;
    size_t buf_sz = sizeof(vsi_nn_internal_node_param_t) + size;
    void* ptr = NULL;
    if( !inode )
    {
        return ptr;
    }

    param = (vsi_nn_internal_node_param_t *)malloc(buf_sz);
    if( param )
    {
        memset( param, 0x00, buf_sz );
        ptr = (void *)(&param->param[0]);
        LINKLIST_APPEND(inode->param, param);
    }

    return ptr;
} /* vsi_nn_internal_new_node_param() */

vsi_nn_internal_tensor_t* vsi_nn_internal_new_tensor
    (
    vsi_nn_node_t*          node,
    vsi_nn_tensor_attr_t*   attr,
    float                   default_value
    )
{
    vsi_nn_internal_tensor_t* tensor = NULL;

    tensor = vsi_nn_internal_create_tensor( node->graph,
                attr, default_value );
    if( tensor )
    {
        LINKLIST_APPEND( WKSP(node)->tensors, tensor );
    }

    return tensor;
} /* vsi_nn_internal_new_tensor() */

vsi_status vsi_nn_internal_release_node
    (
    vsi_nn_internal_node_t** node
    )
{
    if( node && *node )
    {
        vsi_nn_internal_node_t* ptr = *node;

        if( ptr->inputs && ptr->node->input.num )
        {
            free( ptr->inputs );
            ptr->inputs = NULL;
        }
        if( ptr->outputs && ptr->node->output.num )
        {
            free( ptr->outputs );
            ptr->outputs = NULL;
        }
        if( ptr->param )
        {
            vsi_nn_LinkListDeinit((vsi_nn_link_list_t *)(ptr->param), NULL);
        }
        if( ptr->node )
        {
            vsi_nn_ReleaseNode( &ptr->node );
        }

        free( ptr );
        *node = NULL;
    }

    return VSI_SUCCESS;
} /* vsi_nn_internal_release_node() */

vsi_status vsi_nn_internal_release_tensor
    (
    vsi_nn_internal_tensor_t** tensor
    )
{
    if( tensor && *tensor )
    {
        vsi_nn_internal_tensor_t* ptr = *tensor;

        if( ptr->t )
        {
            vsi_nn_ReleaseTensor( &ptr->t );
        }
        free( ptr );
        *tensor = NULL;
    }

    return VSI_SUCCESS;
} /* vsi_nn_internal_release_tensor() */

vsi_bool vsi_nn_internal_check_node
    (
    vsi_nn_internal_node_t* inode
    )
{
    vsi_bool retn = TRUE;

    retn = vsi_nn_OpCheck( inode->node->op, inode->node, inode->inputs, inode->outputs );

    return retn;
} /* vsi_nn_internal_setup_node() */

vsi_bool vsi_nn_internal_setup_node
    (
    vsi_nn_node_t* node,
    vsi_nn_internal_node_t* inode
    )
{
    vsi_bool retn = TRUE;

    retn = vsi_nn_OpSetup( inode->node->op, inode->node, inode->inputs, inode->outputs );
    if( retn )
    {
        inode->node->uid = WKSP(node)->curr_node_uid;
        LINKLIST_APPEND( WKSP(node)->nodes, inode );
        WKSP(node)->curr_node_uid++;

        retn = vsi_nn_internal_check_node(inode);
    }

    return retn;
} /* vsi_nn_internal_setup_node() */

static vsi_status _set_reference_tensor_name
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t* node,
    vsi_nn_node_t* sub_node,
    vsi_nn_tensor_t ** outputs
    )
{
#define _NODE_ID_LEN 64
    vsi_status status;
    vsi_nn_tensor_t *tensor;
    uint32_t i;
    char name[_NODE_ID_LEN];
    if (NULL == node || NULL == graph)
    {
        return VSI_FAILURE;
    }

    status = VSI_SUCCESS;
    for (i = 0; i < sub_node->output.num; i++)
    {
        memset(name, 0, sizeof(char) * _NODE_ID_LEN);
        snprintf(name, sizeof(char) * _NODE_ID_LEN, "uid_%u_sub_uid_%u_out_%u", node->uid, sub_node->uid, i);
        tensor = outputs[i];
        if (tensor && tensor->t)
        {
            status = vxSetReferenceName((vx_reference)tensor->t, name);
            TEST_CHECK_STATUS(status, final);
        }
    }

final:
    return status;
} /* _set_reference_tensor_name() */

vsi_status vsi_nn_internal_compute_node
    (
    vsi_nn_node_t* node
    )
{
    vsi_status status =  VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;
    uint32_t j = 0;

    curr = WKSP(node)->nodes;
    while( NULL != curr )
    {
        for ( j = 0; j < curr->node->output.num; j++ )
        {
            if( NULL == curr->outputs[j] || NULL != curr->outputs[j]->t )
                continue;
            vsi_nn_TensorReinit( node->graph, curr->outputs[j] );
        }

        VSILOGD("Compute node uid[%u] sub_uid[%u] op[%s]",
            node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

        status = _set_reference_tensor_name(node->graph, node, curr->node, curr->outputs);
        if ( VSI_SUCCESS != status )
        {
            VSILOGW("Set reference node[%d] sub_uid[%u] %s output tensor name fail",
                node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));
        }
        status = vsi_nn_OpCompute( curr->node->op, curr->node, curr->inputs, curr->outputs );
        if( VSI_SUCCESS != status )
        {
            VSILOGE("op_compute fail %d", curr->node->op);
            break;
        }

        status = vsi_nn_update_node_attr(curr->node);
        if( VSI_SUCCESS != status )
        {
            VSILOGW("Update node attribute fail");
        }

        curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
    }

    return status;
} /* vsi_nn_internal_compute_node() */

vsi_status vsi_nn_internal_optimize_node
    (
    vsi_nn_node_t* node,
    vsi_nn_opt_direction_e direction
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_internal_node_t* curr = NULL;
    int32_t n = 0;

    curr = WKSP(node)->nodes;
    n = (int32_t)vsi_nn_LinkListGetNodeNumber((vsi_nn_link_list_t *)WKSP(node));

    if (direction == VSI_NN_OPTIMIZE_BACKWARD)
    {
        int32_t i = 0;

        for ( i = n - 1; i >= 0; i-- )
        {
            curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListGetIndexNode((vsi_nn_link_list_t *)WKSP(node), i);
            VSILOGD("Optimize backward for node uid[%u] sub_uid[%u] op[%s]",
                node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

            status = vsi_nn_OpOptimize( curr->node->op, curr->node,
                curr->inputs, curr->outputs, direction );
            if ( VSI_SUCCESS != status )
            {
                VSILOGE("op_optimize backward fail %d", curr->node->op);
                break;
            }

        }
    }
    else
    {
        while( NULL != curr )
        {
            VSILOGD("Optimize forward for node uid[%u] sub_uid[%u] op[%s]",
                node->uid, curr->node->uid, vsi_nn_OpGetName(curr->node->op));

            status = vsi_nn_OpOptimize( curr->node->op, curr->node,
                curr->inputs, curr->outputs, direction );
            if( VSI_SUCCESS != status )
            {
                VSILOGE("op_optimize forward fail %d", curr->node->op);
                break;
            }

            curr = (vsi_nn_internal_node_t *)vsi_nn_LinkListNext( (vsi_nn_link_list_t *)curr );
        }
    }

    return status;
} /* vsi_nn_internal_optimize_node() */

/* EOF */
