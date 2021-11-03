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
#ifndef _VSI_NN_HASHMAP_H
#define _VSI_NN_HASHMAP_H

#include <stdint.h>
#include "vsi_nn_types.h"
#include "vsi_nn_link_list.h"

#if defined(__cplusplus)
extern "C"{
#endif

#define VSI_NN_HASHMAP_KEY_SIZE    (21)
typedef struct
{
    vsi_nn_link_list_t link_list;
    char             * hash_key;
    void             * data;
} vsi_nn_hashmap_item_t;

typedef struct
{
    vsi_nn_hashmap_item_t   * items;
    void                    * values;
    size_t                    size;
} vsi_nn_hashmap_t;

vsi_nn_hashmap_t * vsi_nn_hashmap_create();

void vsi_nn_hashmap_release
    ( vsi_nn_hashmap_t ** map_ptr );

void vsi_nn_hashmap_clear
    ( vsi_nn_hashmap_t * map );

void* vsi_nn_hashmap_get
    (
    const vsi_nn_hashmap_t  * map,
    const char              * key
    );

size_t vsi_nn_hashmap_get_size( const vsi_nn_hashmap_t * map );

void vsi_nn_hashmap_add
    (
    vsi_nn_hashmap_t  * map,
    const char        * key,
    void              * value
    );

void vsi_nn_hashmap_remove
    (
    vsi_nn_hashmap_t    * map,
    const char          * key
    );

vsi_bool vsi_nn_hashmap_has
    (
    vsi_nn_hashmap_t    * map,
    const char          * key
    );

vsi_nn_hashmap_item_t* vsi_nn_hashmap_iter
    ( vsi_nn_hashmap_t* map, vsi_nn_hashmap_item_t* item );

#if defined(__cplusplus)
}
#endif

#endif
