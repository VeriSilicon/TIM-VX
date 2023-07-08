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
#ifndef _VSI_NN_LINK_LIST_H
#define _VSI_NN_LINK_LIST_H

#include "vsi_nn_types.h"

#if defined(__cplusplus)
extern "C"{
#endif

#define vsi_nn_LinkListInitRoot(n) {n = NULL;}

typedef struct _vsi_nn_link_list
{
    struct _vsi_nn_link_list * prev;
    struct _vsi_nn_link_list * next;
} VSI_PUBLIC_TYPE vsi_nn_link_list_t;

typedef void ( * vsi_nn_link_list_init_t )
    (
    vsi_nn_link_list_t * node
    );

typedef void ( * vsi_nn_link_list_deinit_t )
    (
    vsi_nn_link_list_t * node
    );

OVXLIB_API vsi_nn_link_list_t * vsi_nn_LinkListPopStart
    (
    vsi_nn_link_list_t ** root
    );

OVXLIB_API vsi_nn_link_list_t * vsi_nn_LinkListPopEnd
    (
    vsi_nn_link_list_t ** root
    );

OVXLIB_API void vsi_nn_LinkListPushStart
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * nodes
    );

OVXLIB_API void vsi_nn_LinkListPushEnd
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * nodes
    );

OVXLIB_API vsi_nn_link_list_t * vsi_nn_LinkListNext
    (
    vsi_nn_link_list_t * iter
    );

OVXLIB_API vsi_nn_link_list_t * vsi_nn_LinkListNewNode
    (
    size_t sz,
    vsi_nn_link_list_init_t init
    );

OVXLIB_API void vsi_nn_LinkListRemoveNode
    (
    vsi_nn_link_list_t ** root,
    vsi_nn_link_list_t  * nodes
    );

OVXLIB_API void vsi_nn_LinkListDeinit
    (
    vsi_nn_link_list_t * root,
    vsi_nn_link_list_deinit_t deinit
    );

OVXLIB_API vsi_nn_link_list_t *vsi_nn_LinkListGetIndexNode
    (
    vsi_nn_link_list_t * root,
    uint32_t index
    );

OVXLIB_API void vsi_nn_LinkListDelIndexNode
    (
    vsi_nn_link_list_t ** root,
    uint32_t index
    );

OVXLIB_API uint32_t vsi_nn_LinkListGetNodeNumber
    (
    vsi_nn_link_list_t * root
    );

#if defined(__cplusplus)
}
#endif

#endif
