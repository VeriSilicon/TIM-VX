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
#include "utils/vsi_nn_link_list.h"
#include "utils/vsi_nn_hashmap.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_types.h"

typedef struct _tree
{
    struct _tree * left;
    struct _tree * right;
    const char *   hash_key;
    void * data_ptr;
} _binary_tree_t;

static _binary_tree_t * _new_node
    ( void )
{
    _binary_tree_t * node;

    node = (_binary_tree_t *)malloc(
        sizeof( _binary_tree_t ) );

    if (node)
    {
        memset( node, 0, sizeof( _binary_tree_t ) );
    }

    return node;
} /* _new_node() */

static _binary_tree_t * _min_node
    (
    _binary_tree_t * node
    )
{
    _binary_tree_t * cur = node;

    while( NULL != cur->left )
    {
        cur = cur->left;
    }

    return cur;
} /* _min_node() */

static _binary_tree_t * _del_node_by_key
    (
    _binary_tree_t * root,
    const char * hash_key
    )
{
    if( NULL == root )
    {
        return root;
    }
    if( strcmp( hash_key, root->hash_key) < 0 )
    {
        root->left = _del_node_by_key( root->left, hash_key );
    }
    else if( strcmp( hash_key, root->hash_key) > 0 )
    {
        root->right = _del_node_by_key( root->right, hash_key );
    }
    else
    {
        if( NULL == root->left )
        {
            _binary_tree_t * node = root->right;
            free( root );
            return node;
        }
        else if( NULL == root->right )
        {
            _binary_tree_t * node = root->left;
            free( root );
            return node;
        }
        else
        {
            _binary_tree_t * node = _min_node( root->right );
            root->hash_key = node->hash_key;

            /* copy data */
            root->data_ptr = node->data_ptr;

            root->right = _del_node_by_key(root->right, node->hash_key );
        }
    }

    return root;
} /* _del_node_by_key() */

static _binary_tree_t * _move_left
    (
    _binary_tree_t * node
    )
{
    _binary_tree_t * left = NULL;
    if( NULL != node )
    {
        left = node->left;
    }
    return left;
} /* _move_left() */

static _binary_tree_t * _move_right
    (
    _binary_tree_t * node
    )
{
    _binary_tree_t * right = NULL;
    if( NULL != node )
    {
        right = node->right;
    }
    return right;
} /* _move_right() */

static _binary_tree_t * _find_loc
    (
    _binary_tree_t * node,
    const char * hash_key,
    int * val
    )
{
    int tmp;
    _binary_tree_t * next;
    _binary_tree_t * loc;
    loc = NULL;
    tmp = 0;
    while( NULL != node )
    {
        if( strcmp( node->hash_key, hash_key ) > 0 )
        {
            next = _move_left( node );
            tmp = -1;
        }
        else if( strcmp( node->hash_key, hash_key ) < 0 )
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


void _binary_tree_remove_node
    (
    _binary_tree_t ** root,
    const char * hash_key
    )
{
    if( !root || NULL == *root )
    {
        return;
    }

    *root = _del_node_by_key( *root, hash_key );
} /* _binary_tree_remove_node() */

void _binary_tree_new_node
    (
    _binary_tree_t ** root,
    const char * hash_key,
    void * data
    )
{
    int val;
    _binary_tree_t * iter;
    _binary_tree_t * node;

    if( NULL == root )
    {
        return;
    }

    val = 0;
    iter = *root;
    iter = _find_loc( iter, hash_key, &val );
    if( NULL != iter && strcmp( hash_key, iter->hash_key) == 0 )
    {
        VSILOGD( "Key %s has been registered, update value.", hash_key );
        // Update node
        iter->data_ptr = data;
        return;
    }

    /* New node */
    node = _new_node();
    if( NULL != node )
    {
        node->hash_key = hash_key;
        node->data_ptr = data;
    }
    else
    {
        VSILOGW( "Malloc binary tree node fail." );
    }

    /* Insert node */
    if( NULL == iter )
    {
        if( *root == NULL)
        {
            *root = node;
        }
        else
        {
            // Root must be NULL.
            VSI_ASSERT( FALSE );
            free( node );
        }
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
} /* _binary_tree_new_node() */

void * _binary_tree_get_node
    (
    _binary_tree_t ** root,
    const char * hash_key
    )
{
    void * data;
    _binary_tree_t * iter;

    if( NULL == root )
    {
        return NULL;
    }
    data = NULL;
    iter = *root;
    iter = _find_loc( iter, hash_key, NULL );
    if( NULL != iter && strcmp( hash_key, iter->hash_key ) == 0 )
    {
        data = iter->data_ptr;
    }
    return data;
} /* _binary_tree_get_node() */

static void _free_item( vsi_nn_hashmap_item_t * item )
{
    free( item->hash_key );
    free( item );
} /* _free_item() */

vsi_nn_hashmap_t * vsi_nn_hashmap_create()
{
    vsi_nn_hashmap_t * map;
    map = (vsi_nn_hashmap_t*)malloc( sizeof(vsi_nn_hashmap_t) );
    if( NULL == map )
    {
        VSILOGE("Out of memory, create hashmap fail.");
        return NULL;
    }
    memset( map, 0, sizeof( vsi_nn_hashmap_t ) );
    return map;
} /* vsi_nn_hashmap_create() */

void vsi_nn_hashmap_release
    ( vsi_nn_hashmap_t ** map_ptr )
{
    if( map_ptr && *map_ptr )
    {
        vsi_nn_hashmap_clear( *map_ptr );
        free( *map_ptr );
        *map_ptr = NULL;
    }
} /* vsi_nn_hashmap_release() */

void vsi_nn_hashmap_clear( vsi_nn_hashmap_t * map )
{
    if( map )
    {
        vsi_nn_hashmap_item_t * iter = map->items;
        vsi_nn_hashmap_item_t * next = NULL;

        while( NULL != iter )
        {
            next = (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNext(
                    (vsi_nn_link_list_t *)iter );
            _binary_tree_remove_node( (_binary_tree_t**)&(map->values), iter->hash_key );
            vsi_nn_LinkListRemoveNode( (vsi_nn_link_list_t **)&map->items,
                    (vsi_nn_link_list_t *)iter );
            _free_item( (vsi_nn_hashmap_item_t*)iter );
            iter = next;
        }
    }
}

void* vsi_nn_hashmap_get
    (
    const vsi_nn_hashmap_t  * map,
    const char              * key
    )
{
    const char * hash_key = key;
    if( NULL == map )
    {
        return NULL;
    }
    return _binary_tree_get_node( (_binary_tree_t**)&map->values, hash_key );
} /* vsi_nn_hashmap_get() */

void vsi_nn_hashmap_add
    (
    vsi_nn_hashmap_t  * map,
    const char        * key,
    void              * value
    )
{
    vsi_nn_hashmap_item_t * iter;
    size_t key_size = 0;
    const char * hash_key = key;
    if( NULL == map )
    {
        return;
    }
    if( NULL == key )
    {
        return;
    }
    iter = map->items;
    while( NULL != iter )
    {
        if( strcmp( iter->hash_key, hash_key ) == 0 )
        {
            break;
        }
        iter = (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNext(
                (vsi_nn_link_list_t *)iter );
    }
    if( NULL == iter )
    {
        iter = (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNewNode(
                sizeof( vsi_nn_hashmap_item_t ), NULL );
        VSI_ASSERT( iter );
        key_size = strlen( hash_key ) + 1;
        iter->hash_key = (char*)malloc( sizeof(char) * key_size );
        VSI_ASSERT( iter->hash_key );
        memcpy( iter->hash_key, hash_key, key_size );
        vsi_nn_LinkListPushStart( (vsi_nn_link_list_t **)&map->items,
                (vsi_nn_link_list_t *)iter );
        map->size += 1;
    }
    iter->data = value;
    _binary_tree_new_node( (_binary_tree_t**)&map->values, iter->hash_key, value );
} /* vsi_nn_hashmap_add() */

void vsi_nn_hashmap_remove
    (
    vsi_nn_hashmap_t    * map,
    const char          * key
    )
{
    vsi_nn_hashmap_item_t * iter;
    const char * hash_key = key;
    if( NULL == map )
    {
        return;
    }
    _binary_tree_remove_node( (_binary_tree_t**)&(map->values), hash_key );
    iter = map->items;
    while( NULL != iter )
    {
        if( strcmp( iter->hash_key, hash_key ) == 0 )
        {
            break;
        }
        iter = (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNext(
                (vsi_nn_link_list_t *)iter );
    }
    if( NULL != iter )
    {
        vsi_nn_LinkListRemoveNode( (vsi_nn_link_list_t **)&map->items,
                (vsi_nn_link_list_t *)iter );
        _free_item( (vsi_nn_hashmap_item_t*)iter );
        map->size -= 1;
    }
} /* vsi_nn_hashmap_remove() */

vsi_bool vsi_nn_hashmap_has
    (
    vsi_nn_hashmap_t    * map,
    const char          * key
    )
{
    const char * hash_key = key;
    if( NULL == map )
    {
        return FALSE;
    }
    if( NULL == _binary_tree_get_node( (_binary_tree_t**)&map->values, hash_key ) )
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}  /* vsi_nn_hashmap_has() */

size_t vsi_nn_hashmap_get_size( const vsi_nn_hashmap_t * map )
{
    if( !map )
    {
        return 0;
    }
    return map->size;
} /* vsi_nn_hashmap_get_size() */

vsi_nn_hashmap_item_t* vsi_nn_hashmap_iter
    ( vsi_nn_hashmap_t* map, vsi_nn_hashmap_item_t* item )
{
    if( !map )
    {
        return NULL;
    }
    if( !item )
    {
        return map->items;
    }
    return (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNext((vsi_nn_link_list_t *)item );
} /* vsi_nn_hashmap_iter() */

