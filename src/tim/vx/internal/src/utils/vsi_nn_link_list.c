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
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_link_list.h"
#include "vsi_nn_types.h"
#include "vsi_nn_error.h"

static vsi_nn_link_list_t * _walk_to_start
    (
    vsi_nn_link_list_t * root
    );

static vsi_nn_link_list_t * _walk_to_end
    (
    vsi_nn_link_list_t * root
    );

static vsi_nn_link_list_t * _move_next
    (
    vsi_nn_link_list_t * root
    );

static vsi_nn_link_list_t * _move_prev
    (
    vsi_nn_link_list_t * root
    );

static vsi_nn_link_list_t * _walk_to_start
    (
    vsi_nn_link_list_t * root
    )
{
    if( NULL != root )
    {
        while( NULL != root->prev )
        {
            root = root->prev;
        }
    }
    return root;
} /* _walk_to_start() */

static vsi_nn_link_list_t * _walk_to_end
    (
    vsi_nn_link_list_t * root
    )
{
    if( NULL != root )
    {
        while( NULL != root->next )
        {
            root = root->next;
        }
    }
    return root;
} /* _walk_to_end() */

static vsi_nn_link_list_t * _move_next
    (
    vsi_nn_link_list_t * root
    )
{
    if( NULL != root )
    {
        root = root->next;
    }
    return root;
} /* _move_next() */

static vsi_nn_link_list_t * _move_prev
    (
    vsi_nn_link_list_t * root
    )
{
    if( NULL != root )
    {
        root = root->prev;
    }
    return root;
} /* _move_prev() */

vsi_nn_link_list_t * vsi_nn_LinkListPopStart
    (
    vsi_nn_link_list_t ** root
    )
{
    vsi_nn_link_list_t * node;
    vsi_nn_link_list_t * self;

    if( NULL == root || NULL == *root )
    {
        return NULL;
    }

    node = NULL;
    self = *root;

    self = _walk_to_start( self );
    node = self;

    self = _move_next( self );

    if( NULL != self )
    {
        self->prev = NULL;
    }
    if( NULL != node )
    {
        node->next = NULL;
    }

    *root = self;
    return node;
} /* vsi_nn_LinkListPopStart() */

vsi_nn_link_list_t * vsi_nn_LinkListPopEnd
    (
    vsi_nn_link_list_t ** root
    )
{
    vsi_nn_link_list_t * node;
    vsi_nn_link_list_t * self;

    if( NULL == root || NULL == *root )
    {
        return NULL;
    }

    node = NULL;
    self = *root;

    self = _walk_to_end( self );
    node = self;

    self = _move_prev( self );

    if( NULL != self )
    {
        self->next = NULL;
    }
    if( NULL != node )
    {
        node->prev = NULL;
    }

    *root = self;
    return node;
} /* vsi_nn_LinkListPopEnd() */

void vsi_nn_LinkListPushStart
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * nodes
    )
{
    vsi_nn_link_list_t * self;
    if( NULL == root || NULL == nodes )
    {
        return;
    }
    if( NULL == *root )
    {
        *root = nodes;
    }
    else if( NULL != nodes )
    {
        self = *root;
        self = _walk_to_start( self );
        nodes = _walk_to_end( nodes );
        nodes->next = self;
        self->prev = nodes;
        self = _walk_to_start( self );
        *root = self;
    }
} /* vsi_nn_LinkListPushStart() */

void vsi_nn_LinkListPushEnd
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * nodes
    )
{
    vsi_nn_link_list_t * self;
    if( NULL == root || NULL == nodes )
    {
        return;
    }
    if( NULL == *root )
    {
        *root = nodes;
    }
    else if( NULL != nodes )
    {
        self = *root;
        self = _walk_to_end( self );
        nodes = _walk_to_start( nodes );
        nodes->prev = self;
        self->next = nodes;
        self = _walk_to_start( self );
        *root = self;
    }
} /* vsi_nn_LinkListPushEnd() */

vsi_nn_link_list_t * vsi_nn_LinkListNext
    (
    vsi_nn_link_list_t * iter
    )
{
    return _move_next( iter );
} /* vsi_nn_LinkListNext() */

vsi_nn_link_list_t * vsi_nn_LinkListNewNode
    (
    size_t sz,
    vsi_nn_link_list_init_t init
    )
{
    vsi_nn_link_list_t *node = (vsi_nn_link_list_t *)malloc(sz);
    CHECK_PTR_FAIL_GOTO( node, "Create node fail.", final );
    memset(node, 0, sz);

    if(init)
    {
        init(node);
    }

final:
    return node;
} /* vsi_nn_LinkListNewNode() */

void vsi_nn_LinkListRemoveNode
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * node
    )
{
    vsi_nn_link_list_t * iter;
    iter = *root;
    iter = _walk_to_start( iter );
    while( NULL != iter )
    {
        if( iter == node )
        {
            break;
        }
        iter = _move_next( iter );
    }
    if( NULL != iter )
    {
        if( iter == *root )
        {
            if( NULL != iter->prev )
            {
                *root = iter->prev;
            }
            else if( NULL != iter->next )
            {
                *root = iter->next;
            }
            else
            {
                *root = NULL;
            }
        }
        if( NULL != iter->prev )
        {
            iter->prev->next = iter->next;
        }
        if( NULL != iter->next )
        {
            iter->next->prev = iter->prev;
        }
    }
} /* vsi_nn_LinkListRemoveNode() */

void vsi_nn_LinkListDeinit
    (
    vsi_nn_link_list_t * root,
    vsi_nn_link_list_deinit_t deinit
    )
{
    vsi_nn_link_list_t *tmp = NULL;

    while(root)
    {
        tmp = (vsi_nn_link_list_t *)vsi_nn_LinkListPopStart( &root );
        if(tmp)
        {
            if(deinit)
            {
                deinit(tmp);
            }
            free(tmp);
            tmp = NULL;
        }
    }
} /* vsi_nn_LinkListDeinit() */

vsi_nn_link_list_t *vsi_nn_LinkListGetIndexNode
    (
    vsi_nn_link_list_t * root,
    uint32_t index
    )
{
    uint32_t n;
    vsi_nn_link_list_t *iter;
    if(NULL == root)
    {
        return NULL;
    }

    n = 0;
    iter = _walk_to_start(root);
    while (iter)
    {
        if(n == index)
        {
            return iter;
        }
        n++;
        iter = _move_next(iter);
    }
    return NULL;
} /* vsi_nn_LinkListGetIndexNode() */

void vsi_nn_LinkListDelIndexNode
    (
    vsi_nn_link_list_t ** root,
    uint32_t index
    )
{
    uint32_t n;
    vsi_nn_link_list_t *iter;
    if(NULL == root || NULL == *root)
    {
        return ;
    }

    n = 0;
    iter = _walk_to_start(*root);
    while (iter)
    {
        if(n == index)
        {
            vsi_nn_link_list_t *del = iter;
            if(iter->prev == NULL && iter->next == NULL) /* Only one node */
            {
                *root = NULL;
            }
            else if(iter->prev == NULL && iter->next != NULL) /* head */
            {
                iter = _move_next(iter);
                iter->prev = NULL;
                *root = iter;
            }
            else if(iter->prev != NULL && iter->next == NULL) /* tail */
            {
                iter = _move_prev(iter);
                iter->next = NULL;
            }
            else
            {
                iter->prev->next = iter->next;
                iter->next->prev = iter->prev;
                iter = _move_next(iter);
            }
            free(del);
            return ;
        }
        n++;
        iter = _move_next(iter);
    }
}

uint32_t vsi_nn_LinkListGetNodeNumber
    (
    vsi_nn_link_list_t * root
    )
{
    uint32_t n;
    vsi_nn_link_list_t *iter;
    if(NULL == root)
    {
        return 0;
    }

    n = 0;
    iter = _walk_to_start(root);
    while (iter)
    {
        iter = _move_next(iter);
        n++;
    }

    return n;
}
