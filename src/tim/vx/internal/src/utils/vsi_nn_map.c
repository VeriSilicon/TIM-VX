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
#include "utils/vsi_nn_binary_tree.h"
#include "utils/vsi_nn_map.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "vsi_nn_types.h"

void vsi_nn_MapInit
    (
    vsi_nn_map_t * map
    )
{
    if( NULL == map )
    {
        return;
    }
    memset( map, 0, sizeof( vsi_nn_map_t ) );
} /* vsi_nn_MapInit() */

void * vsi_nn_MapGet
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    )
{
    if( NULL == map )
    {
        return NULL;
    }
    return vsi_nn_BinaryTreeGetNode( &map->values, key );
} /* vsi_nn_MapGet() */

void vsi_nn_MapAdd
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key,
    void              * value
    )
{
    vsi_nn_map_key_list_t * key_iter;
    if( NULL == map )
    {
        return;
    }
    vsi_nn_BinaryTreeNewNode( &map->values, key, value );
    key_iter = map->keys;
    while( NULL != key_iter )
    {
        if( key_iter->val == key )
        {
            break;
        }
        key_iter = (vsi_nn_map_key_list_t *)vsi_nn_LinkListNext(
                (vsi_nn_link_list_t *)key_iter );
    }
    if( NULL == key_iter )
    {
        key_iter = (vsi_nn_map_key_list_t *)vsi_nn_LinkListNewNode(
                sizeof( vsi_nn_map_key_list_t ), NULL );
        key_iter->val = key;
        vsi_nn_LinkListPushStart( (vsi_nn_link_list_t **)&map->keys,
                (vsi_nn_link_list_t *)key_iter );
        map->size += 1;
    }
} /* vsi_nn_MapAdd() */

void vsi_nn_MapRemove
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    )
{
    vsi_nn_map_key_list_t * key_iter;
    if( NULL == map )
    {
        return;
    }
    vsi_nn_BinaryTreeRemoveNode( &map->values, key );
    key_iter = map->keys;
    while( NULL != key_iter )
    {
        if( key_iter->val == key )
        {
            break;
        }
        key_iter = (vsi_nn_map_key_list_t *)vsi_nn_LinkListNext(
                (vsi_nn_link_list_t *)key_iter );
    }
    if( NULL != key_iter )
    {
        vsi_nn_LinkListRemoveNode( (vsi_nn_link_list_t **)&map->keys,
                (vsi_nn_link_list_t *)key_iter );
        free( key_iter );
        map->size -= 1;
    }
} /* vsi_nn_MapRemove() */

vsi_bool vsi_nn_MapHasKey
    (
    vsi_nn_map_t      * map,
    vsi_nn_map_key_t    key
    )
{
    if( NULL == map )
    {
        return FALSE;
    }
    if( NULL == vsi_nn_BinaryTreeGetNode( &map->values, key ) )
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}  /* vsi_nn_MapHasKey() */

