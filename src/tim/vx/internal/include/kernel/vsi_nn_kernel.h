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

#ifndef _VSI_NN_KERNEL_H
#define _VSI_NN_KERNEL_H

#include <stdint.h>
#include "vsi_nn_log.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_daemon.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_util.h"
#include "utils/vsi_nn_shape_util.h"
#include "utils/vsi_nn_hashmap.h"
#include "utils/vsi_nn_math.h"
#include "kernel/vsi_nn_gpu.h"
#include "libnnext/vx_lib_nnext.h"

__BEGIN_DECLS

/** Kernel types */
typedef enum
{
    VSI_NN_KERNEL_TYPE_CPU = 0,
    VSI_NN_KERNEL_TYPE_EVIS,
    VSI_NN_KERNEL_TYPE_CL,
    VSI_NN_KERNEL_TYPE_VX,
    VSI_NN_KERNEL_TYPE_SP,
    VSI_NN_KERNEL_TYPE_NUM,
    VSI_NN_KERNEL_TYPE_NONE = VSI_NN_KERNEL_TYPE_NUM
} VSI_PUBLIC_TYPE  vsi_nn_kernel_type_e;

/** Kernel pirority */
enum
{
    VSI_NN_KERNEL_PIRORITY_DISABLE = 0,
    VSI_NN_KERNEL_PIRORITY_NORMAL_LIMIT = 0x1FFFFFFF,
    VSI_NN_KERNEL_PIRORITY_FORCE_EXEC = 0x20000000,
};

/** Kernel internal data type */
typedef enum
{
    I8 = 0,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    BF16,
    BOOL8,
    I4,
    U4,
} VSI_PUBLIC_TYPE vsi_nn_kernel_dtype_e;

typedef enum
{
    VSI_NN_KERNEL_QUANT_NONE,
    VSI_NN_KERNEL_QUANT_DFP,
    VSI_NN_KERNEL_QUANT_ASYMM,
    VSI_NN_KERNEL_QUANT_ASYMM_PERCHANNEL,
    VSI_NN_KERNEL_QUANT_SYMM,
    VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL,
    VSI_NN_KERNEL_QUANT_TYPE_NUM
} vsi_nn_kernel_quant_type_e;

/** GPU source format */
typedef enum
{
    VSI_NN_GPU_SOURCE_FMT_CODE = 0,
    VSI_NN_GPU_SOURCE_FMT_EXECUTABLE = 1,
    VSI_NN_GPU_SOURCE_FMT_NUM
} VSI_PUBLIC_TYPE vsi_nn_gpu_source_fmt_e;

typedef char * vsi_nn_kernel_source_t;
typedef uint32_t vsi_nn_kernel_unique_id_t;

typedef struct
{
    char * data;
} vsi_nn_kernel_build_option_t;

typedef struct
{
   size_t num;
   vsi_nn_kernel_source_t * data;
   vsi_nn_kernel_build_option_t build_option;
} vsi_nn_kernel_source_info_t;

typedef struct
{
    vsi_nn_kernel_type_e      type;
    vsi_nn_kernel_unique_id_t unique_id;
    vx_kernel_description_t   info;
    struct
    {
        vsi_nn_kernel_source_info_t sources[VSI_NN_GPU_SOURCE_FMT_NUM];
        vsi_nn_gpu_source_fmt_e active_source_fmt;
    } gpu;
} VSI_PUBLIC_TYPE vsi_nn_kernel_t;

typedef struct
{
    int32_t fl;
} vsi_nn_kernel_quant_dfp_t;

typedef struct
{
    float   scale;
    int32_t zero_point;
} vsi_nn_kernel_quant_asymm_t;

typedef struct
{
    vsi_float_array_t * scale;
    vsi_int_array_t   * zero_point;
    int32_t             channel_dim;
} vsi_nn_kernel_quant_asymm_perchannel_t;

typedef struct
{
    vsi_nn_kernel_dtype_e       dtype;
    vsi_size_array_t           * shape;
    vsi_nn_kernel_quant_type_e  quant;
    union
    {
        vsi_nn_kernel_quant_dfp_t dfp;
        vsi_nn_kernel_quant_asymm_t asymm;
        vsi_nn_kernel_quant_asymm_perchannel_t asymm_v;
    };
    float scale;
    int32_t zero_point;
} vsi_nn_kernel_tensor_attr_t;

typedef struct
{
    vsi_nn_kernel_type_e kernel_type;
    int32_t fps;
} vsi_nn_kernel_pirority_t;

typedef struct
{
    vsi_nn_kernel_pirority_t pirority[VSI_NN_KERNEL_TYPE_NUM];
    int32_t allow_kernel_num;
} vsi_nn_kernel_selector_t;

typedef void * VSI_PUBLIC_TYPE vsi_nn_kernel_node_param_t;

typedef void * vsi_nn_kernel_tensor_t;

typedef void * VSI_PUBLIC_TYPE vsi_nn_kernel_node_t;

typedef void * vsi_nn_kernel_graph_t;

typedef void * VSI_PUBLIC_TYPE vsi_nn_kernel_scalar_t;

typedef vsi_nn_hashmap_t vsi_nn_kernel_param_t;

typedef vsi_nn_kernel_node_t (* vsi_nn_kernel_setup_func_t)
    (
    vsi_nn_graph_t *,
    vsi_nn_tensor_t **,
    size_t input_num,
    vsi_nn_tensor_t **,
    size_t output_num,
    const vsi_nn_kernel_param_t *,
    vsi_nn_kernel_t *
    );

typedef vsi_status (* vsi_nn_kernel_selector_func_t)
    (
    vsi_nn_graph_t *,
    vsi_nn_tensor_t **,
    size_t input_num,
    vsi_nn_tensor_t **,
    size_t output_num,
    const vsi_nn_kernel_param_t *,
    vsi_nn_kernel_selector_t *
    );

typedef struct
{
    vsi_nn_kernel_unique_id_t  unique_id;
    vsi_nn_kernel_setup_func_t setup[VSI_NN_KERNEL_TYPE_NUM];
    vsi_nn_kernel_selector_func_t select;
} vsi_nn_kernel_backend_t;

vsi_nn_kernel_param_t * vsi_nn_kernel_param_create();

void vsi_nn_kernel_param_release( vsi_nn_kernel_param_t ** params );

void vsi_nn_kernel_param_clear( vsi_nn_kernel_param_t * params );

vsi_bool vsi_nn_kernel_param_add_int32
    ( vsi_nn_kernel_param_t * params, const char * key, int32_t value);

int32_t vsi_nn_kernel_param_get_int32
    ( const vsi_nn_kernel_param_t * params, const char * key);

vsi_bool vsi_nn_kernel_param_add_int64
    ( vsi_nn_kernel_param_t * params, const char * key, int64_t value);

int64_t vsi_nn_kernel_param_get_int64
    ( const vsi_nn_kernel_param_t * params, const char * key);

vsi_bool vsi_nn_kernel_param_add_float32
    ( vsi_nn_kernel_param_t * params, const char * key, float value);

float vsi_nn_kernel_param_get_float32
    ( const vsi_nn_kernel_param_t * params, const char * key);

vsi_bool vsi_nn_kernel_param_add_str
    ( vsi_nn_kernel_param_t * params, const char * key, const char * str);

const char * vsi_nn_kernel_param_get_str
    ( const vsi_nn_kernel_param_t * params, const char * key);

vsi_bool vsi_nn_kernel_param_add_buffer
    ( vsi_nn_kernel_param_t * params, const char * key, void * buf, size_t size);

void * vsi_nn_kernel_param_get_buffer
    ( const vsi_nn_kernel_param_t * params, const char * key, size_t * size);

vsi_bool vsi_nn_kernel_param_add_const_buffer
    ( vsi_nn_kernel_param_t * params, const char * key, const void * buf, size_t size);

const void * vsi_nn_kernel_param_get_const_buffer
    ( const vsi_nn_kernel_param_t * params, const char * key, size_t * size);

/** Kernel register */
#define REGISTER_KERNEL_BACKEND(kernel_name, kernel_type, func)   \
        _INITIALIZER(_register_kernel_##kernel_name##_##kernel_type) \
        { \
            vsi_nn_kernel_backend_register( \
                    ""#kernel_name, \
                    VSI_NN_KERNEL_TYPE_##kernel_type, func ); \
        }
#define REGISTER_KERNEL_SELECTOR(kernel_name, func) \
        _INITIALIZER(_register_kernel_##kernel_name##_selector) \
        { \
            vsi_nn_kernel_selector_register( \
                    ""#kernel_name, func ); \
        }

#if 0
    typedef struct
    {
        const char* name;
        vsi_nn_op_t op;
        vsi_nn_kernel_type_e kernel_type;
        vsi_nn_kernel_setup_func_t func;
    } vsi_nn_kernel_section_meta_t;
    #define REGISTER_KERNEL_BACKEND(operation, kernel_type, func) \
        static vsi_nn_kernel_section_meta_t _kernel_meta = \
                {""#operation, VSI_NN_OP_##operation, VSI_NN_KERNEL_TYPE_##kernel_type, func}; \
        static vsi_nn_kernel_section_meta_t* _kernel_meta_ptr \
            __attribute__((section("kernel_meta_section"))) = &_kernel_meta;
#endif
#if 0
    #define REGISTER_KERNEL_BACKEND(operation, kernel_type, func)   \
                vsi_status func##_(vsi_nn_graph_t* graph, \
                        vsi_nn_tensor_t** inputs, size_t input_num, \
                        vsi_nn_tensor_t** outputs, size_t output_num) {\
                    return func(graph, inputs, input_num,  outputs, output_num); \
                }

    #define REGISTER_KERNEL_BACKEND_MANUALLY(operation, kernel_type, func) \
                extern vsi_status func##_(vsi_nn_graph_t*, \
                        vsi_nn_tensor_t** inputs, size_t input_num, \
                        vsi_nn_tensor_t** outputs, size_t output_num); \
                vsi_nn_kernel_backend_register( ""#operation, \
                        VSI_NN_KERNEL_TYPE_##kernel_type, func##_ );
#endif

#define REGISTER_BACKEND_CL(operation, func) \
    REGISTER_KERNEL_BACKEND(operation, CL, func)
#define REGISTER_BACKEND_EVIS(operation, func) \
    REGISTER_KERNEL_BACKEND(operation, EVIS, func)
#define REGISTER_BACKEND_CPU(operation, func) \
    REGISTER_KERNEL_BACKEND(operation, CPU, func)
#define REGISTER_BACKEND_OPENVX(operation, func) \
    REGISTER_KERNEL_BACKEND(operation, VX, func)
#define REGISTER_BACKEND_STREAM_PROCESSOR(operation, func) \
    REGISTER_KERNEL_BACKEND(operation, SP, func)

#define DEF_KERNEL_BASE_CALLBACK( NAME )  \
    static vsi_status NAME##_impl( vsi_nn_kernel_node_t node, \
            const vsi_nn_kernel_node_param_t * param, \
            size_t param_size ); \
    static vx_status VX_CALLBACK NAME( \
            vx_node node, const vx_reference * param,\
            vx_uint32 param_size) {\
                return (vx_status)NAME##_impl( \
                        (vsi_nn_kernel_node_t)node, \
                        (const vsi_nn_kernel_node_param_t *)param, \
                        (uint32_t)param_size \
                        ); \
            } \
    static vsi_status NAME##_impl

#define DEF_SP_KERNEL_BASE_CALLBACK( NAME )  \
    static vsi_status NAME##_impl( vsi_nn_kernel_node_t node); \
    static vx_status VX_CALLBACK NAME( \
            vx_node node) {\
                return (vx_status)NAME##_impl( \
                        (vsi_nn_kernel_node_t)node); \
            } \
    static vsi_status NAME##_impl


#define DEF_KERNEL_INITIALIZER( NAME )          DEF_KERNEL_BASE_CALLBACK( NAME )
#define DEF_KERNEL_EXECUTOR( NAME )             DEF_KERNEL_BASE_CALLBACK( NAME )
#define DEF_KERNEL_DEINITIALIZER( NAME )        DEF_KERNEL_BASE_CALLBACK( NAME )
#define DEF_SP_KERNEL_QUERY( NAME )             DEF_SP_KERNEL_BASE_CALLBACK( NAME )

void vsi_nn_kernel_backend_register
    (
    const char * kernel_name,
    vsi_nn_kernel_type_e kernel_type,
    vsi_nn_kernel_setup_func_t setup_func
    );

const vsi_nn_kernel_backend_t * vsi_nn_kernel_backend_get
    ( const char * );

vsi_status vsi_nn_kernel_backend_init( void );

void vsi_nn_kernel_backend_deinit( void );

void vsi_nn_kernel_selector_register
    (
    const char * kernel_name,
    vsi_nn_kernel_selector_func_t selecotr_func
    );

vsi_status vsi_nn_kernel_pirority_set
    (
    vsi_nn_kernel_selector_t * selector,
    const vsi_nn_kernel_pirority_t * pirority,
    size_t pirority_size
    );

vsi_nn_kernel_t * vsi_nn_kernel_create
    (
    vsi_nn_kernel_type_e type
    );

void vsi_nn_kernel_reset
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_kernel_type_e type
    );

void vsi_nn_kernel_release
    (
    vsi_nn_kernel_t ** kernel
    );

void vsi_nn_kernel_add_source
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_gpu_source_fmt_e fmt,
    size_t source_num,
    ...
    );

void vsi_nn_kernel_add_build_option
    (
    vsi_nn_kernel_t * kernel,
    const char * option
    );

vsi_nn_kernel_tensor_t vsi_nn_kernel_tensor_create
    (
    vsi_nn_kernel_graph_t graph,
    const vsi_nn_kernel_tensor_attr_t* attr,
    vsi_bool is_virtual
    );

void vsi_nn_kernel_tensor_release
    (
    vsi_nn_kernel_tensor_t * tensor
    );

vsi_nn_kernel_tensor_t vsi_nn_kernel_tensor_reshape
    (
    vsi_nn_kernel_tensor_t tensor,
    vsi_size_t * shape,
    vsi_size_t rank
    );

vsi_status vsi_nn_kernel_node_pass_param
    (
    vsi_nn_kernel_node_t node,
    vsi_nn_kernel_node_param_t * params,
    size_t num
    );

static VSI_INLINE_API void vsi_nn_kernel_node_release
    (
    vsi_nn_kernel_node_t * node
    )
{
    if( node && *node )
    {
        vxReleaseNode( (vx_node*)node );
    }
}

static VSI_INLINE_API void vsi_nn_kernel_node_pack_io
    (
    vsi_nn_kernel_node_param_t * params,
    size_t param_num,
    vsi_nn_tensor_t ** inputs,
    size_t input_num,
    vsi_nn_tensor_t ** outputs,
    size_t output_num
    )
{
    size_t i;
    size_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < input_num && cnt < param_num; i ++, cnt ++ )
    {
        if( inputs[i] )
        {
            params[cnt] = (vsi_nn_kernel_node_param_t)(inputs[i]->t);
        }
        else
        {
            params[cnt] = NULL;
        }
    }

    /* Set outputs */
    for( i = 0; i < output_num && cnt < param_num; i ++, cnt ++ )
    {
        if( outputs[i] )
        {
            params[cnt] = (vsi_nn_kernel_node_param_t)(outputs[i]->t);
        }
        else
        {
            params[cnt] = NULL;
        }
    }
} /* vsi_nn_kernel_node_pack_io() */

/** Kernel selector */
vsi_nn_kernel_node_t vsi_nn_kernel_selector
    (
    vsi_nn_graph_t * graph,
    const char * kernel_name,
    vsi_nn_tensor_t ** inputs,
    size_t input_num,
    vsi_nn_tensor_t ** outputs,
    size_t output_num,
    const vsi_nn_kernel_param_t * params
    );

/** Map data type to gpu internal dtype. */
static VSI_INLINE_API vsi_nn_kernel_dtype_e vsi_nn_kernel_map_dtype
    (
    vsi_nn_type_e dtype
    )
{
    switch( dtype )
    {
    case VSI_NN_TYPE_INT4:
        return I4;
    case VSI_NN_TYPE_UINT4:
        return U4;
    case VSI_NN_TYPE_INT8:
        return I8;
    case VSI_NN_TYPE_BOOL8:
        return BOOL8;
    case VSI_NN_TYPE_INT16:
        return I16;
    case VSI_NN_TYPE_INT32:
        return I32;
    case VSI_NN_TYPE_INT64:
        return I64;
    case VSI_NN_TYPE_UINT8:
        return U8;
    case VSI_NN_TYPE_UINT16:
        return U16;
    case VSI_NN_TYPE_UINT32:
        return U32;
    case VSI_NN_TYPE_FLOAT16:
        return F16;
    case VSI_NN_TYPE_BFLOAT16:
        return BF16;
    case VSI_NN_TYPE_FLOAT32:
        return F32;
    default:
        VSILOGE("error data type %d", dtype);
        break;
    }
    return I8;
} /* vsi_nn_kernel_map_dtype() */

static VSI_INLINE_API  vsi_nn_type_e vsi_nn_dtype_map_kernel
    (
    vsi_nn_kernel_dtype_e dtype
    )
{
    switch( dtype )
    {
    case I4:
        return VSI_NN_TYPE_INT4;
    case U4:
        return VSI_NN_TYPE_UINT4;
    case I8:
        return VSI_NN_TYPE_INT8;
    case BOOL8:
        return VSI_NN_TYPE_BOOL8;
    case I16:
        return VSI_NN_TYPE_INT16;
    case I32:
        return VSI_NN_TYPE_INT32;
    case I64:
        return VSI_NN_TYPE_INT64;
    case U8:
        return VSI_NN_TYPE_UINT8;
    case U16:
        return VSI_NN_TYPE_UINT16;
    case U32:
        return VSI_NN_TYPE_UINT32;
    case F16:
        return VSI_NN_TYPE_FLOAT16;
    case BF16:
        return VSI_NN_TYPE_BFLOAT16;
    case F32:
        return VSI_NN_TYPE_FLOAT32;
    default:
        VSILOGE("error data type %d", dtype);
        break;
    }
    return VSI_NN_TYPE_INT8;
} /* vsi_nn_kernel_map_dtype() */

static VSI_INLINE_API size_t vsi_nn_kernel_dtype_get_bytes
    (
    vsi_nn_kernel_dtype_e dtype
    )
{
    switch( dtype )
    {
        case I8:
        case U8:
        case BOOL8:
            return sizeof(int8_t);
        case I16:
        case U16:
        case F16:
        case BF16:
            return sizeof(int16_t);
        case I32:
        case U32:
        case F32:
            return sizeof(int32_t);
        case I64:
            return sizeof(int64_t);
        default:
            VSILOGE("Error data type %d", dtype);
            break;
    }
    return 0;
} /* vsi_nn_kernel_dtype_get_bytes() */

static VSI_INLINE_API vsi_size_t vsi_nn_kernel_dtype_get_bits
    (
    vsi_nn_kernel_dtype_e dtype
    )
{
    switch( dtype )
    {
        case I4:
        case U4:
            return 4;
        case I8:
        case U8:
        case BOOL8:
            return 8;
        case I16:
        case U16:
        case F16:
        case BF16:
            return 16;
        case I32:
        case U32:
        case F32:
            return 32;
        case I64:
            return 64;
        default:
            VSILOGE("Error data type %d", dtype);
            break;
    }
    return 0;
} /* vsi_nn_kernel_dtype_get_bits() */

static VSI_INLINE_API vsi_nn_kernel_quant_type_e vsi_nn_kernel_map_quant_type
    ( vsi_nn_qnt_type_e quant_type )
{
    switch( quant_type )
    {
        case VSI_NN_QNT_TYPE_DFP:
            return VSI_NN_KERNEL_QUANT_DFP;
        case VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC:
            return VSI_NN_KERNEL_QUANT_ASYMM;
        case VSI_NN_QNT_TYPE_AFFINE_PERCHANNEL_SYMMETRIC:
            return VSI_NN_KERNEL_QUANT_SYMM_PERCHANNEL;
        default:
            break;
    }
    return VSI_NN_KERNEL_QUANT_NONE;
} /* vsi_nn_kernel_map_quant_type() */

vsi_nn_kernel_node_t  vsi_nn_kernel_create_node
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel
    );

vsi_nn_kernel_node_t  vsi_nn_kernel_create_node_ext
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    );

vsi_status vsi_nn_kernel_node_set_border
    (vsi_nn_kernel_node_t node,
    vx_border_t* border);

vsi_nn_kernel_scalar_t vsi_nn_kernel_scalar_create
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_dtype_e dtype,
    const void * data
    );

static VSI_INLINE_API void vsi_nn_kernel_scalar_release
    ( vsi_nn_kernel_scalar_t * scalar )
{
    if( scalar && *scalar )
    {
        vxReleaseScalar( (vx_scalar*)scalar );
    }
} /* vsi_nn_kernel_scalar_relase() */

vsi_status vsi_nn_kernel_scalar_read_uint4
    ( vsi_nn_kernel_scalar_t scalar, uint8_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_int4
    ( vsi_nn_kernel_scalar_t scalar, int8_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_int8
    ( vsi_nn_kernel_scalar_t scalar, int8_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_int32
    ( vsi_nn_kernel_scalar_t scalar, int32_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_int64
    ( vsi_nn_kernel_scalar_t scalar, int64_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_uint8
    ( vsi_nn_kernel_scalar_t scalar, uint8_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_uint32
    ( vsi_nn_kernel_scalar_t scalar, uint32_t * out_data );

vsi_status vsi_nn_kernel_scalar_read_float32
    ( vsi_nn_kernel_scalar_t scalar, float * out_data );

vsi_status vsi_nn_kernel_scalar_read_float64
    ( vsi_nn_kernel_scalar_t scalar, double * out_data );

vsi_status vsi_nn_kernel_scalar_write_int8
    ( vsi_nn_kernel_scalar_t scalar, int8_t out_data );

vsi_status vsi_nn_kernel_scalar_write_int32
    ( vsi_nn_kernel_scalar_t scalar, int32_t out_data );

vsi_status vsi_nn_kernel_scalar_write_int64
    ( vsi_nn_kernel_scalar_t scalar, int64_t out_data );

vsi_status vsi_nn_kernel_scalar_write_uint8
    ( vsi_nn_kernel_scalar_t scalar, uint8_t out_data );

vsi_status vsi_nn_kernel_scalar_write_uint32
    ( vsi_nn_kernel_scalar_t scalar, uint32_t out_data );

vsi_status vsi_nn_kernel_scalar_write_float32
    ( vsi_nn_kernel_scalar_t scalar, float out_data );

vsi_status vsi_nn_kernel_scalar_write_float64
    ( vsi_nn_kernel_scalar_t scalar, double out_data );

vsi_status vsi_nn_kernel_scalar_get_dtype
    (
    vsi_nn_kernel_scalar_t scalar,
    vsi_nn_kernel_dtype_e * dtype
    );

vsi_status vsi_nn_kernel_register
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel
    );

vsi_status vsi_nn_kernel_register_ext
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    );

vsi_bool vsi_nn_kernel_gpu_check_shape
    ( const vsi_size_t * shape, vsi_size_t rank );

vsi_status vsi_nn_kernel_gpu_add_param
    (
    vsi_nn_kernel_node_t node,
    const char * param_key,
    void * data
    );

vsi_status vsi_nn_kernel_gpu_config
    (
    vsi_nn_kernel_node_t node,
    const gpu_param_t * gpu_param
    );

vsi_nn_kernel_tensor_attr_t * vsi_nn_kernel_tensor_attr_create
    ( vsi_nn_kernel_tensor_t tensor );

void vsi_nn_kernel_tensor_attr_release
    ( vsi_nn_kernel_tensor_attr_t ** attr );

/*
 * Create a buffer with a copy of tensor data.
 * attr is optional
 */
void * vsi_nn_kernel_tensor_create_buffer
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    vsi_bool convert_to_float
    );

/*
 * Read tensor data to buffer.
 * attr is optional
 */
vsi_status vsi_nn_kernel_tensor_read
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    void * out_buffer,
    size_t out_buffer_size
    );

/*
 * Write float data to tensor.
 * attr is optional
 */
vsi_status vsi_nn_kernel_tensor_write_from_float
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const float * float_buffer,
    size_t size
    );

/*
 * Write data to tensor.
 * attr is optional
 */
vsi_status vsi_nn_kernel_tensor_write
    (
    vsi_nn_kernel_tensor_t tensor,
    const vsi_nn_kernel_tensor_attr_t * attr,
    const void * buffer,
    size_t size
    );

static VSI_INLINE_API vsi_size_t vsi_nn_kernel_tensor_attr_get_size
    ( const vsi_nn_kernel_tensor_attr_t * attr )
{
    if( !attr )
    {
        return 0;
    }
    return vsi_nn_shape_get_size( attr->shape->data, (vsi_size_t)attr->shape->size );
} /* vsi_nn_kernel_tensor_attr_get_size() */

static VSI_INLINE_API vsi_size_t vsi_nn_kernel_tensor_attr_get_bytes
    ( const vsi_nn_kernel_tensor_attr_t * attr )
{
    vsi_size_t i = 0;
    vsi_size_t bytes;
    vsi_size_t bits_num;
    vsi_size_t * shape = NULL;
    if( !attr )
    {
        return 0;
    }

    shape = attr->shape->data;

    bits_num = vsi_nn_kernel_dtype_get_bits( attr->dtype );
    if ( bits_num < BITS_PER_BYTE )
    {
        if (shape[0] % 2 == 0)
        {
            bytes = shape[0] / 2;
        }
        else
        {
            bytes = shape[0] / 2 + shape[0] % 2;
        }
    }
    else
    {
        bytes = shape[0] * bits_num / BITS_PER_BYTE;
    }
    for ( i = 1; i < (vsi_size_t)attr->shape->size; i ++ )
    {
        bytes *= shape[i];
    }

    return bytes;
} /* vsi_nn_kernel_tensor_attr_get_bytes() */

static VSI_INLINE_API void vsi_nn_kernel_tensor_attr_get_stride
    ( const vsi_nn_kernel_tensor_attr_t * attr, vsi_size_t * out_stride)
{
    vsi_size_t type_bits;
    vsi_size_t total_bytes;
    vsi_size_t * shape = NULL;

    if( !attr || !out_stride )
    {
        return;
    }

    shape = attr->shape->data;
    type_bits = vsi_nn_kernel_dtype_get_bits( attr->dtype );

    if ( type_bits < BITS_PER_BYTE )
    {
        vsi_size_t i;

        out_stride[0] = type_bits / BITS_PER_BYTE;
        total_bytes = out_stride[0];

        total_bytes = 1;
        if ( shape[0] % (BITS_PER_BYTE / type_bits) == 0 )
        {
             out_stride[1] = shape[0] * type_bits / BITS_PER_BYTE;
        }
        else
        {
             out_stride[1] = shape[0] * type_bits / BITS_PER_BYTE + 1;
        }

        total_bytes *= out_stride[1];
        for (i = 2; i < (vsi_size_t)attr->shape->size; i++)
        {
            out_stride[i] = shape[i - 1] * out_stride[i - 1];
            total_bytes *= shape[i];
        }
        total_bytes *= shape[1];

        for( i = (vsi_size_t)attr->shape->size; i < VSI_NN_MAX_DIM_NUM; i ++ )
        {
            out_stride[i] = total_bytes;
        }
    }
    else
    {
        vsi_nn_shape_get_stride( attr->shape->data, (vsi_size_t)attr->shape->size, out_stride );
    }
} /* vsi_nn_kernel_tensor_attr_get_size() */

static VSI_INLINE_API vsi_bool vsi_nn_kernel_tensor_attr_is_quantized
    ( const vsi_nn_kernel_tensor_attr_t * attr )
{
    return ( attr && attr->quant > VSI_NN_KERNEL_QUANT_NONE
            && attr->quant < VSI_NN_KERNEL_QUANT_TYPE_NUM
            && attr->dtype != F16
            && attr->dtype != BF16
            && attr->dtype != F32
            && attr->dtype != F64 );
} /* vsi_nn_kernel_tensor_attr_is_quantized() */

//TODO: Make vsi_nn_kernel_dtype_e to public and move dtype functions to vsi_nn_dtype.h
vsi_bool vsi_nn_dtype_convert_float_to_dtype
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    void * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_asymm
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    void * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_dfp
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    int32_t fl,
    void * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    void * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_float_to_quantize_symm_perchannel
    (
    const float * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    const vsi_size_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    void * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_dtype_to_float
    (
    const void * buffer,
    size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_asymm_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_dfp_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    int32_t fl,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    float scale, int32_t zero_point,
    float * out_buffer
    );

vsi_bool vsi_nn_dtype_convert_quantize_symm_perchannel_to_float
    (
    const void * buffer, size_t size,
    vsi_nn_kernel_dtype_e dtype,
    const vsi_size_t * shape, size_t rank,
    const float * scale, size_t scale_size,
    const int32_t * zero_point, size_t zero_point_size,
    int32_t channel_dim,
    float * out_buffer
    );

vsi_nn_tensor_t* vsi_nn_pad_tensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_size_t * pad_front,
    vsi_size_t * pad_end,
    vsi_size_t pad_size,
    vsi_nn_pad_mode_e mode,
    float pad_value
    );

vsi_nn_tensor_t* vsi_nn_merge_input_zeropoint_to_bias
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias
    );

void vsi_nn_kernel_add_source_internal
    (
        vsi_nn_kernel_t * kernel,
        vsi_nn_gpu_source_fmt_e fmt,
        size_t source_num,
        va_list args
    );

OVXLIB_API vsi_nn_kernel_t * vsi_nn_KernelCreate
    (
    vsi_nn_kernel_type_e type
    );

OVXLIB_API void vsi_nn_KernelAddSource
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_gpu_source_fmt_e fmt,
    size_t source_num,
    ...
    );

OVXLIB_API void vsi_nn_KernelAddBuildOption
    (
    vsi_nn_kernel_t * kernel,
    const char * option
    );

OVXLIB_API vsi_status vsi_nn_KernelNodePassParam
    (
    vsi_nn_kernel_node_t node,
    vsi_nn_kernel_node_param_t * params,
    size_t num
    );

OVXLIB_API vsi_nn_kernel_node_t  vsi_nn_KernelCreateNodeExt
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_t * kernel,
    const char** resources
    );

OVXLIB_API vsi_nn_kernel_scalar_t vsi_nn_kernelScalarCreate
    (
    vsi_nn_graph_t * graph,
    vsi_nn_kernel_dtype_e dtype,
    const void * data
    );

OVXLIB_API vsi_status vsi_nn_KernelGpuConfig
    (
    vsi_nn_kernel_node_t node,
    const gpu_param_t * gpu_param
    );

static VSI_INLINE_API const char* vsi_nn_kernel_type_str
    (
    vsi_nn_kernel_type_e type
    )
{
    switch( type )
    {
    case VSI_NN_KERNEL_TYPE_CPU:
        return "CPU";
    case VSI_NN_KERNEL_TYPE_EVIS:
        return "EVIS";
    case VSI_NN_KERNEL_TYPE_CL:
        return "CL";
    case VSI_NN_KERNEL_TYPE_VX:
        return "OPENVX";
    case VSI_NN_KERNEL_TYPE_SP:
        return "STERAM_PROCESSOR";
    default:
        break;
    }
    return "None";
} /* vsi_nn_kernel_type_str() */

static VSI_INLINE_API vsi_status vsi_nn_kernel_unpack_4bit_data
    (
    const vsi_nn_kernel_tensor_attr_t * attr,
    uint8_t * src,
    uint8_t * dest,
    vsi_nn_kernel_dtype_e dtype
    )
{
    vsi_status status;
    uint32_t i = 0, j = 0;
    uint8_t high = 0, low = 0;
    vsi_size_t stride[VSI_NN_MAX_DIM_NUM] = {0};
    vsi_size_t src_size;

    status = VSI_SUCCESS;
    vsi_nn_kernel_tensor_attr_get_stride( attr, stride );

    src_size = stride[attr->shape->size];

    for ( i = 0 ; i < src_size; i++)
    {
        high = src[i] >> 4;
        low = src[i] & 0x0F;
        if ( dtype == I4 )
        {
            if( high > 7)
            {
                high = high | 0xF0;
            }
            if( low > 7)
            {
                low = low | 0xF0;
            }
        }
        if ( attr->shape->data[0] % stride[1] == 0 )
        {
            if ( attr->shape->data[0] == 1 )
            {
                dest[j] = low;
                j++;
            }
            else
            {
                dest[j] = low;
                dest[j+1] = high;
                j += 2;
            }
        }
        else
        {
            if ( (i+1) % stride[1] == 0 )
            {
                dest[j] = low;
                j++;
            }
            else
            {
                dest[j] = low;
                dest[j+1] = high;
                j += 2;
            }
        }
    }

    return status;
}

static VSI_INLINE_API vsi_status vsi_nn_kernel_pack_4bit_data
    (
    const vsi_nn_kernel_tensor_attr_t * attr,
    uint8_t * src,
    uint8_t * dest
    )
{
    vsi_status status;
    uint32_t i = 0, j = 0;
    uint8_t high = 0, low = 0;
    vsi_size_t src_size;

    status = VSI_SUCCESS;
    src_size = vsi_nn_kernel_tensor_attr_get_size( attr );
    for ( i = 0; i < src_size; i++ )
    {
        if ( (i+1) % attr->shape->data[0] == 0)
        {
            high = 0;
            low = src[i];
        }
        else
        {
            high = src[i+1];
            low = src[i];
            i++;
        }
        dest[j] = (high << 4) | (low & 0xF);
        j++;
    }

    return status;
}

__END_DECLS

#endif
