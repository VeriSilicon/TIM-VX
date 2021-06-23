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

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "vsi_nn_prv.h"
#include "vsi_nn_types.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_log.h"
#include "vsi_nn_error.h"
#include "vsi_nn_context.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_hashmap.h"

typedef enum
{
    _PARAM_I32 = 1,
    _PARAM_I64,
    _PARAM_F32,
    _PARAM_BUFFER,
    _PARAM_CONST_BUFFER,
    _PARAM_STR,
} _param_dtype_e;

typedef struct
{
    _param_dtype_e type;
    union
    {
        int32_t     int32;
        int64_t     int64;
        float       float32;
        void*       buffer;
        const void* const_buffer;
        const char* str;
    } value;
    size_t size;
} _param_type;

#define CHECK_PARAM_NULL( ptr, rval, ... ) \
    do { \
        if( ptr == NULL ) { \
            VSILOGE(__VA_ARGS__); \
            VSI_ASSERT(FALSE); \
            return rval; \
        } \
    } while(0)

#define _PARAM_ADD_TEMPLATE(TYPE_NAME, TYPE, PARAM_DTYPE) \
    vsi_bool vsi_nn_kernel_param_add_##TYPE_NAME \
        (vsi_nn_kernel_param_t* params, const char* key, TYPE value) \
    { \
        _param_type* p; \
        CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." ); \
        CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." ); \
        p = malloc( sizeof(_param_type) ); \
        CHECK_PARAM_NULL( p, FALSE, "Out of memory, add param fail." ); \
        p->type = PARAM_DTYPE; \
        p->value.TYPE_NAME = value; \
        p->size = sizeof( TYPE ); \
        vsi_nn_hashmap_add( params, key, p ); \
        return TRUE; \
    }
#define _PARAM_GET_TEMPLATE(TYPE_NAME, TYPE, DEFAULT_VALUE, PARAM_DTYPE) \
    TYPE vsi_nn_kernel_param_get_##TYPE_NAME \
        ( const vsi_nn_kernel_param_t* params, const char* key) \
    { \
        _param_type* p; \
        CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." ); \
        CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." ); \
        p = vsi_nn_hashmap_get( params, key ); \
        if( p->type != PARAM_DTYPE ) { \
            VSILOGW("Key %s is not \"%s\"", key, ""#TYPE_NAME ); \
        } \
        CHECK_PARAM_NULL( p, DEFAULT_VALUE, "Key %s not in params.", key ); \
        return p->value.TYPE_NAME; \
    }

_PARAM_ADD_TEMPLATE(int32, int32_t, _PARAM_I32)
_PARAM_ADD_TEMPLATE(int64, int64_t, _PARAM_I64)
_PARAM_ADD_TEMPLATE(float32, float, _PARAM_F32)
_PARAM_GET_TEMPLATE(int32, int32_t, 0, _PARAM_I32)
_PARAM_GET_TEMPLATE(int64, int64_t, 0, _PARAM_I64)
_PARAM_GET_TEMPLATE(float32, float, 0.0f, _PARAM_F32)
_PARAM_GET_TEMPLATE(str, const char*, NULL, _PARAM_STR)

vsi_bool vsi_nn_kernel_param_add_str
    (
    vsi_nn_kernel_param_t * params,
    const char * key,
    const char * value
    )
{
    _param_type* p;
    CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." );
    CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." );
    p = malloc( sizeof(_param_type) );
    CHECK_PARAM_NULL( p, FALSE, "Out of memory, add param fail." );
    p->type = _PARAM_STR;
    p->value.str = value;
    p->size = strlen( value );
    vsi_nn_hashmap_add( params, key, p );
    return TRUE;
} /* vsi_nn_kernel_param_add_str() */

vsi_bool vsi_nn_kernel_param_add_buffer
    (
    vsi_nn_kernel_param_t * params,
    const char * key,
    void * value,
    size_t size
    )
{
    _param_type* p;
    CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." );
    CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." );
    p = malloc( sizeof(_param_type) );
    CHECK_PARAM_NULL( p, FALSE, "Out of memory, add param fail." );
    p->type = _PARAM_BUFFER;
    p->value.buffer = value;
    p->size = size;
    vsi_nn_hashmap_add( params, key, p );
    return TRUE;
} /* vsi_nn_kernel_param_add_buffer() */

void* vsi_nn_kernel_param_get_buffer
    ( const vsi_nn_kernel_param_t * params, const char * key, size_t * size)
{
    _param_type* p;
    CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." );
    CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." );
    p = vsi_nn_hashmap_get( params, key );
    CHECK_PARAM_NULL( p, 0, "Key %s not in params.", key );
    if( p->type != _PARAM_BUFFER )
    {
        VSILOGW("Key %s is not \"buffer\"", key );
    }
    if( size != NULL )
    {
        *size = p->size;
    }
    return p->value.buffer;
} /* vsi_nn_kernel_param_get_buffer() */

vsi_bool vsi_nn_kernel_param_add_const_buffer
    (
    vsi_nn_kernel_param_t * params,
    const char * key,
    const void * value,
    size_t size
    )
{
    _param_type* p;
    CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." );
    CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." );
    p = malloc( sizeof(_param_type) );
    CHECK_PARAM_NULL( p, FALSE, "Out of memory, add param fail." );
    p->type = _PARAM_CONST_BUFFER;
    p->value.const_buffer = value;
    p->size = size;
    vsi_nn_hashmap_add( params, key, p );
    return TRUE;
} /* vsi_nn_kernel_param_add_const_buffer() */

const void* vsi_nn_kernel_param_get_const_buffer
    ( const vsi_nn_kernel_param_t * params, const char * key, size_t * size)
{
    _param_type* p;
    CHECK_PARAM_NULL( params, FALSE, "Params is null ptr." );
    CHECK_PARAM_NULL( key, FALSE, "Param key is null ptr." );
    p = vsi_nn_hashmap_get( params, key );
    CHECK_PARAM_NULL( p, 0, "Key %s not in params.", key );
    if( p->type != _PARAM_CONST_BUFFER )
    {
        VSILOGW("Key %s is not \"const buffer\"", key );
    }
    if( size != NULL )
    {
        *size = p->size;
    }
    return p->value.const_buffer;
} /* vsi_nn_kernel_param_get_const_buffer() */

vsi_nn_kernel_param_t* vsi_nn_kernel_param_create()
{
    return (vsi_nn_kernel_param_t*)vsi_nn_hashmap_create();
} /* vsi_nn_kernel_param_create() */

void vsi_nn_kernel_param_release( vsi_nn_kernel_param_t ** params )
{
    if( params && *params )
    {
        vsi_nn_kernel_param_clear( *params );
        vsi_nn_hashmap_release( (vsi_nn_hashmap_t**)params );
        *params = NULL;
    }
} /* vsi_nn_kernel_param_release() */

void vsi_nn_kernel_param_clear( vsi_nn_kernel_param_t * params )
{
    if( params )
    {
        vsi_nn_hashmap_t* hashmap = (vsi_nn_hashmap_t*)(params);
        vsi_nn_hashmap_item_t* p = vsi_nn_hashmap_iter( hashmap, NULL );
        vsi_nn_hashmap_item_t* next;
        while( p )
        {
            next = vsi_nn_hashmap_iter( hashmap, p );
            free( p->data );
            p = next;
        }
        vsi_nn_hashmap_clear( hashmap );
    }
} /* vsi_nn_kernel_param_clear() */

