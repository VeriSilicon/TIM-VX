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
#include "vsi_nn_ops.h"
#include "vsi_nn_client_op.h"
#include "vsi_nn_node.h"
#include "vsi_nn_types.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"

#define DEF_OP(NAME, ...) extern vsi_nn_op_proc_t vsi_nn_op_##NAME;
#include "interface/ops.def"
#undef DEF_OP
#define DEF_OP(NAME, ...) &vsi_nn_op_##NAME,
static const vsi_nn_op_proc_t * vsi_nn_ops_tab[VSI_NN_OP_NUM] =
{
#include "interface/ops.def"
};
#undef DEF_OP

/* Custom ops */
#define DEF_OP(NAME, ...) extern vsi_nn_op_proc_t vsi_nn_op_##NAME;
#include "custom/custom_ops.def"
#undef DEF_OP
#define DEF_OP(NAME, ...) &vsi_nn_op_##NAME,
static const vsi_nn_op_proc_t * vsi_nn_custom_ops_tab[VSI_NN_OP_CUSTOM_NUM] =
{
#include "custom/custom_ops.def"
};
#undef DEF_OP

/* Internal ops */
#define DEF_OP(NAME, ...) extern vsi_nn_op_proc_t vsi_nn_op_##NAME;
#include "internal/internal_ops.def"
#undef DEF_OP
#define DEF_OP(NAME, ...) &vsi_nn_op_##NAME,
static const vsi_nn_op_proc_t * vsi_nn_internal_ops_tab[VSI_NN_OP_INTERNAL_NUM] =
{
#include "internal/internal_ops.def"
};
#undef DEF_OP

// TODO: Add name item to op structure
#define DEF_OP(NAME, ...) ""#NAME,
static const char * vsi_nn_ops_name[] =
{
#include "interface/ops.def"
    "UNKNOWN"
};
#undef DEF_OP

#define DEF_OP(NAME, ...) ""#NAME,
static const char * vsi_nn_custom_ops_name[] =
{
#include "custom/custom_ops.def"
    "UNKNOWN"
};
#undef DEF_OP

// TODO: Add name item to internal op structure
#define DEF_OP(NAME, ...) ""#NAME,
static const char * vsi_nn_internal_ops_name[] =
{
#include "internal/internal_ops.def"
    "UNKNOWN"
};
#undef DEF_OP


vsi_bool _is_external_ops(vsi_nn_op_t op) {
    vsi_bool ret = FALSE;
    if (op <  0) {
        ret = TRUE;
    }
    return ret;
}

vsi_bool _is_custom_ops
    (
    vsi_nn_op_t op
    )
{
    vsi_bool ret = FALSE;
    if( op > VSI_NN_OP_CUSTOM_START && op < VSI_NN_OP_CUSTOM_END )
    {
        ret = TRUE;
    }
    return ret;
}

vsi_bool _is_internal_ops
    (
    vsi_nn_op_t op
    )
{
    vsi_bool ret = FALSE;
    if( op > VSI_NN_OP_INTERNAL_START && op < VSI_NN_OP_INTERNAL_END )
    {
        ret = TRUE;
    }
    return ret;
}

vsi_bool vsi_nn_OpIsValid
    (
    vsi_nn_op_t op
    )
{
    vsi_bool valid;
    valid = TRUE;

    if( (op < VSI_NN_OP_NUM ) || _is_internal_ops(op) )
    {
        valid = TRUE;
    }
    else
    {
        if( _is_custom_ops(op) == FALSE )
        {
            valid = vsi_nn_OpIsRegistered( op );
        }
    }
    return valid;
} /* vsi_nn_OpIsValid() */

const vsi_nn_op_proc_t * vsi_nn_OpGetProc
    (
    vsi_nn_op_t op
    )
{
    const vsi_nn_op_proc_t * proc;

    /* Use client first */
    proc = vsi_nn_OpGetClient( op );
    if( NULL == proc && op < VSI_NN_OP_NUM )
    {
        proc = vsi_nn_ops_tab[op];
    }
    if( NULL == proc && _is_custom_ops(op) )
    {
        proc = vsi_nn_custom_ops_tab[op - VSI_NN_OP_CUSTOM_START - 1];
    }
    if( NULL == proc && _is_internal_ops(op) )
    {
        proc = vsi_nn_internal_ops_tab[op - VSI_NN_OP_INTERNAL_START - 1];
    }
    return proc;
} /* vsi_nn_OpGetProc() */

vsi_status vsi_nn_OpInit
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_status status;

    status = VSI_FAILURE;
    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc )
    {
        status = VSI_SUCCESS;
        if( NULL != proc->init )
        {
            status = proc->init( node );
        }
    }
    return status;
} /* vsi_nn_OpInit() */

vsi_status vsi_nn_OpCompute
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_status status;

    status = VSI_FAILURE;
    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc )
    {
        if( NULL != proc->compute )
        {
            status = proc->compute( node, inputs, outputs );
        }
        else
        {
            VSILOGE("Do not support this platform");
            status = VSI_FAILURE;
        }
    }
    return status;
} /* vsi_nn_op_compute() */

vsi_status vsi_nn_OpDeinit
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_status status;

    status = VSI_FAILURE;
    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc )
    {
        status = VSI_SUCCESS;
        if( proc->deinit )
        {
        status = proc->deinit( node );
        }
    }
    return status;
} /* vsi_nn_op_deinit() */

vsi_status vsi_nn_OpOptimize
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_status status;

    status = VSI_SUCCESS;
    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc && proc->optimize != NULL )
    {
        status = proc->optimize( node, inputs, outputs, direction );
        if(status != VSI_SUCCESS)
        {
            VSILOGE("Optimize node %s fail", vsi_nn_OpGetName(node->op));
        }
    }
    return status;
} /* vsi_nn_OpOptimize() */

vsi_bool vsi_nn_OpCheck
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_bool ret;

    ret = FALSE;
    proc = vsi_nn_OpGetProc( op );
    if ( NULL != proc )
    {
        ret = TRUE;
        if ( proc->check && node->graph->ctx->options.enable_opcheck)
        {
            ret = proc->check( node, inputs, outputs );
        }
    }
    return ret;
} /* vsi_nn_op_check() */

void vsi_nn_OpGetIoNum
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_size_t     * input_num,
    vsi_size_t     * output_num
    )
{
    const vsi_nn_op_proc_t * proc;

    VSI_UNREFERENCED(node);

    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc )
    {
        if( NULL != input_num )
        {
            *input_num = (uint32_t)proc->input_num;
        }
        if( NULL != output_num )
        {
            *output_num = (uint32_t)proc->output_num;
        }
    }
} /* vsi_nn_OpGetIoNum() */

vsi_bool vsi_nn_OpGenerateTensor
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret;
    ret = vsi_nn_OpSetup( node->op, node, inputs, outputs );
    return ret;
} /* vsi_nn_OpGenerateTensor() */

vsi_bool vsi_nn_OpSetup
    (
    vsi_nn_op_t op,
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_bool ret;

    ret = FALSE;
    proc = vsi_nn_OpGetProc( op );
    if( NULL != proc )
    {
        ret = proc->setup( node, inputs, outputs );
    }
    return ret;
} /* vsi_nn_OpSetup() */

vsi_bool vsi_nn_OpRegisterOvxInit
    (
    vsi_nn_op_t op,
    vsi_nn_op_compute_t compute
    )
{
    const vsi_nn_op_proc_t * proc;
    vsi_nn_op_proc_t tmp;
    vsi_bool ret;

    ret = FALSE;
    proc = vsi_nn_OpGetProc( op );

    if( NULL != proc )
    {
        memcpy( &tmp, proc, sizeof( vsi_nn_op_proc_t ) );
        tmp.compute = compute;
        vsi_nn_OpRegisterClient( op, &tmp );
    }
    return ret;
} /* vsi_nn_OpRegisterClientCompute() */

vsi_bool vsi_nn_OpRegisterExternalOvxInit
    (
    vsi_nn_op_t op,
    const char* kernel_name,
    vsi_nn_op_proc_t* proc
    )
{
    vsi_bool ret;

    ret = FALSE;
    if (vsi_nn_OpRegisterClient(op, proc) &&
        vsi_nn_OpAddClientName(op, kernel_name)) {
        ret = TRUE;
    }
    return ret;
}

const char * vsi_nn_OpGetName
    (
    vsi_nn_op_t op
    )
{
    const char * name;
    if(_is_external_ops(op)){
        name = vsi_nn_OpGetClientName(op);
    }
    else if( op < VSI_NN_OP_NUM )
    {
        name = vsi_nn_ops_name[op];
    }
    else if(_is_custom_ops(op))
    {
        name = vsi_nn_custom_ops_name[op - VSI_NN_OP_CUSTOM_START - 1];
    }
    else if(_is_internal_ops(op))
    {
        name = vsi_nn_internal_ops_name[op - VSI_NN_OP_INTERNAL_START - 1];
    }
    else
    {
        name = vsi_nn_ops_name[VSI_NN_OP_NUM];
    }
    return name;
} /* vsi_nn_GetOpName() */
