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
#include <stdlib.h>
#include "utils/vsi_nn_binary_tree.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"

static vsi_nn_binary_tree_t * _new_node
    ( void )
{
    vsi_nn_binary_tree_t * node;

    node = (vsi_nn_binary_tree_t *)malloc(
        sizeof( vsi_nn_binary_tree_t ) );
    if (node)
    {
        memset( node, 0, sizeof( vsi_nn_binary_tree_t ) );
    }

    return node;
} /* _new_node() */

static vsi_nn_binary_tree_t * _min_node
    (
    vsi_nn_binary_tree_t * node
    )
{
    vsi_nn_binary_tree_t * cur = node;

    while( NULL != cur->left )
    {
        cur = cur->left;
    }

    return cur;
} /* _min_node() */

static vsi_nn_binary_tree_t * _del_node_by_key
    (
    vsi_nn_binary_tree_t * root,
    vsi_nn_binary_tree_key_t key
    )
{
    if( NULL == root )
    {
        return root;
    }
    if( key < root->key )
    {
        root->left = _del_node_by_key( root->left, key );
    }
    else if( key > root->key )
    {
        root->right = _del_node_by_key( root->right, key );
    }
    else
    {
        if( NULL == root->left )
        {
            vsi_nn_binary_tree_t * node = root->right;
            free( root );
            return node;
        }
        else if( NULL == root->right )
        {
            vsi_nn_binary_tree_t * node = root->left;
            free( root );
            return node;
        }
        else
        {
            vsi_nn_binary_tree_t * node = _min_node( root->right );
            root->key = node->key;

            /* copy data */
            root->data_ptr = node->data_ptr;

            root->right = _del_node_by_key(root->right, node->key );
        }
    }

    return root;
} /* _del_node_by_key() */

static vsi_nn_binary_tree_t * _move_left
    (
    vsi_nn_binary_tree_t * node
    )
{
    vsi_nn_binary_tree_t * left = NULL;
    if( NULL != node )
    {
        left = node->left;
    }
    return left;
} /* _move_left() */

static vsi_nn_binary_tree_t * _move_right
    (
    vsi_nn_binary_tree_t * node
    )
{
    vsi_nn_binary_tree_t * right = NULL;
    if( NULL != node )
    {
        right = node->right;
    }
    return right;
} /* _move_right() */

static vsi_nn_binary_tree_t * _find_loc
    (
    vsi_nn_binary_tree_t * node,
    vsi_nn_binary_tree_key_t key,
    int * val
    )
{
    int tmp;
    vsi_nn_binary_tree_t * next;
    vsi_nn_binary_tree_t * loc;
    loc = NULL;
    tmp = 0;
    while( NULL != node )
    {
        if( node->key > key )
        {
            next = _move_left( node );
            tmp = -1;
        }
        else if( node->key < key )
        {
            next = _move_right( node );
            tmp = 1;
        }
        else
        {
            loc = node;
            tmp = 0;
            break;
        }
        if( NULL != next )
        {
            node = next;
        }
        else
        {
            loc = node;
            break;
        }
    }
    if( NULL != val )
    {
        *val = tmp;
    }
    return loc;
} /* _find_loc_to_insert() */


void vsi_nn_BinaryTreeRemoveNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key
    )
{
    if ( NULL == root )
    {
        return;
    }

    *root = _del_node_by_key( *root, key );
} /* vsi_nn_BinaryTreeRemoveNode() */

void vsi_nn_BinaryTreeNewNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key,
    void * data
    )
{
    int val;
    vsi_nn_binary_tree_t * iter;
    vsi_nn_binary_tree_t * node;

    if( NULL == root )
    {
        return;
    }

    val = 0;
    iter = *root;
    iter = _find_loc( iter, key, &val );
    if( NULL != iter && key == iter->key )
    {
        //VSILOGE( "Key %#x has been registered.", (unsigned int)key );
        // Update node
        iter->data_ptr = data;
        return;
    }

    /* New node */
    node = _new_node();
    if( NULL != node )
    {
        node->key = key;
        node->data_ptr = data;
    }
    else
    {
        VSILOGW( "Malloc binary tree node fail." );
    }

    /* Insert node */
    if( NULL == iter )
    {
        *root = node;
    }
    else
    {
        if( val > 0 )
        {
            iter->right = node;
        }
        else if( val < 0 )
        {
            iter->left = node;
        }
        else
        {
            VSILOGE( "Hash collision!" );
            if( node )
            {
                free( node );
                node = NULL;
            }
        }
    }
} /* vsi_nn_BinaryTreeNewNode() */

void * vsi_nn_BinaryTreeGetNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key
    )
{
    void * data;
    vsi_nn_binary_tree_t * iter;

    if( NULL == root )
    {
        return NULL;
    }
    data = NULL;
    iter = *root;
    iter = _find_loc( iter, key, NULL );
    if( NULL != iter && key == iter->key )
    {
        data = iter->data_ptr;
    }
    return data;
} /* vsi_nn_BinaryTreeGetNode() */
