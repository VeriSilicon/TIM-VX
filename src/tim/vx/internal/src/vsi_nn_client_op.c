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
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_client_op.h"
#include "utils/vsi_nn_binary_tree.h"


typedef struct _client_node
{
    vsi_nn_op_t                op;
    vsi_nn_op_proc_t           proc;
} _client_node_t;

static vsi_nn_binary_tree_t * s_root = NULL;

static _client_node_t * _create_client_node
    (
    vsi_nn_op_t op,
    vsi_nn_op_proc_t * proc
    )
{
    _client_node_t * node;
    node = (_client_node_t *)malloc( sizeof( _client_node_t ) );
    if( NULL != node )
    {
        node->op = op;
        memcpy( &node->proc, proc, sizeof( vsi_nn_op_proc_t ) );
    }
    return node;
} /* _create_client_node() */

static void _release_client_node
    (
    _client_node_t ** node
    )
{
    if( NULL != node && NULL != *node )
    {
        free( *node );
        *node = NULL;
    }
} /* _release_client_node() */

vsi_bool vsi_nn_OpIsRegistered
    (
    vsi_nn_op_t op
    )
{
    return ( NULL != vsi_nn_OpGetClient( op ) );
} /* vsi_nn_OpIsRegistered() */

vsi_bool vsi_nn_OpRegisterClient
    (
    vsi_nn_op_t op,
    vsi_nn_op_proc_t * proc
    )
{
    vsi_bool ret;
    _client_node_t * node;

    ret = FALSE;
    if( TRUE ==  vsi_nn_OpIsRegistered( op ) )
    {
        VSILOGE( "OP %#x has been registered.", op );
        return ret;
    }

    node = _create_client_node( op, proc );
    if( NULL != node )
    {
        vsi_nn_BinaryTreeNewNode(
            &s_root,
            (vsi_nn_binary_tree_key_t)op,
            (void *)node
            );
        ret = TRUE;
    }
    return ret;
} /* vsi_nn_OpRegisterClient() */

vsi_nn_op_proc_t * vsi_nn_OpGetClient
    (
    vsi_nn_op_t op
    )
{
    vsi_nn_op_proc_t * proc;
    _client_node_t * node;

    proc = NULL;
    node = (_client_node_t *)vsi_nn_BinaryTreeGetNode(
        &s_root,
        (vsi_nn_binary_tree_key_t)op );
    if( NULL != node )
    {
        proc = &node->proc;
    }
    return proc;
} /* vsi_nn_OpGetClient() */

void vsi_nn_OpRemoveClient
    (
    vsi_nn_op_t op
    )
{
    _client_node_t * node;

    node = (_client_node_t *)vsi_nn_BinaryTreeGetNode(
        &s_root,
        (vsi_nn_binary_tree_key_t)op );
    if( NULL != node )
    {
        _release_client_node( &node );
        vsi_nn_BinaryTreeRemoveNode( &s_root, op );
    }
} /* vsi_nn_OpRemoveClient() */

