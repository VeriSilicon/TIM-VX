/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_log.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"

__BEGIN_DECLS

/*
 * Define kernel meta.
 */
#define KERNEL_SOURCE_1    "scatter_nd_update"
#define KERNEL_SOURCE_2    "scatter_nd_update_big"
#define KERNEL_SOURCE_3    "scatter_nd_update_atom"
#define KERNEL_SOURCE_4    "scatter_nd_update_special"
#define KERNEL_SOURCE_5    "scatter_nd_update_qint"
#define KERNEL_SOURCE_6    "scatter_nd_update_fp"

#define HASH_SCATTER_ND_UPDATE_KEY(_in0_type, _in2_type, _out_type, _stage, _coord_type, _opt_flg) \
    ((_in0_type << 24) | (_in2_type << 16) | (_out_type << 8) | (_stage << 4) | (_coord_type << 2) | (_opt_flg))

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_RESET_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_reset_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_UPDATE_NAME(SRC2_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_update_"#SRC2_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_UPDATE_4X_NAME(SRC2_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_update_"#SRC2_TYPE"_4X")

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_REF_NAME(SRC2_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_ref_"#SRC2_TYPE"to"#DST_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_REF_4X_NAME(SRC2_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_ref_"#SRC2_TYPE"to"#DST_TYPE"_4X")

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_COPY_NAME(DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_copy_"#DST_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_REF_NAME(SRC0_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_ref2out_"#SRC0_TYPE"to"#DST_TYPE)

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_UPDATE_NAME(SRC2_TYPE, DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_update2ref_"#SRC2_TYPE"to"#DST_TYPE"_16x")

#define HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_COPY_NAME(DST_TYPE) \
    CVIVANTE_NAMESPACE("evis.scatter_nd_update_cpy2out_"#DST_TYPE"to"#DST_TYPE)

#define TENSOR_SCATTER_ND_UPDATE_SPECIAL_REF_KERNELS(IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, IN2_TYPE, OUT_TYPE, 4, 1, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_REF_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_SPECIAL_UPDATE_KERNELS(IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, IN2_TYPE, OUT_TYPE, 5, 1, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_UPDATE_NAME(IN2_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_SPECIAL_COPY_KERNELS(IN0_TYPE, IN1_TYPE, IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, IN2_TYPE, OUT_TYPE, 6, 1, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_SPECIAL_COPY_NAME(IN0_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(IN0_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(IN0_TYPE, 0, OUT_TYPE, 0, 0, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_RESET_NAME(IN0_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(IN2_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(0, IN2_TYPE, 0, 1, 0, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_UPDATE_NAME(IN2_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(IN2_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(0, IN2_TYPE, 0, 1, 0, 1), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_UPDATE_4X_NAME(IN2_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(0, IN2_TYPE, OUT_TYPE, 2, 0, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_REF_NAME(IN2_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(IN2_TYPE, OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(0, IN2_TYPE, OUT_TYPE, 2, 0, 1), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_REF_4X_NAME(IN2_TYPE, OUT_TYPE), \
        SOURCE },

#define TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(OUT_TYPE, SOURCE) \
    { HASH_SCATTER_ND_UPDATE_KEY(0, 0, OUT_TYPE, 3, 0, 0), \
        HASH_SCATTER_ND_UPDATE_SH_KERNEL_COPY_NAME(OUT_TYPE), \
        SOURCE },

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type scatter_nd_update_reset_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(U8,   U8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(I8,   I8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(I16,  I16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(F16,  F16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(BF16, BF16, KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(U8,   F16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(I8,   F16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(I16,  F16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_RESET_KERNELS(F16,  U8,   KERNEL_SOURCE_5)
};

static const _kernel_map_type scatter_nd_update_update_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(U8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(I8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(I16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_KERNELS(BF16, KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(U8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(I8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(I16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_UPDATE_4X_KERNELS(BF16, KERNEL_SOURCE_6)
};

static const _kernel_map_type scatter_nd_update_ref_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(I32, U8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(I32, I8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(I32, I16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(I32, F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(F32, F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_REF_KERNELS(F32, BF16, KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(I32, U8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(I32, I8,   KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(I32, I16,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(I32, F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(F32, F16,  KERNEL_SOURCE_6)
    TENSOR_SCATTER_ND_UPDATE_REF_4X_KERNELS(F32, BF16, KERNEL_SOURCE_6)
};

static const _kernel_map_type scatter_nd_update_copy_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(U8,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(I8,  KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(I16, KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(F16, KERNEL_SOURCE_5)
    TENSOR_SCATTER_ND_UPDATE_COPY_KERNELS(BF16, KERNEL_SOURCE_5)
};

static const _kernel_map_type scatter_nd_update_special_ref_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_REF_KERNELS(U8,  I32, U8,  U8, KERNEL_SOURCE_4)
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_REF_KERNELS(I8,  I32, I8,  I8, KERNEL_SOURCE_4)
};

static const _kernel_map_type scatter_nd_update_special_update_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_UPDATE_KERNELS(U8,  I32, U8,  U8, KERNEL_SOURCE_4)
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_UPDATE_KERNELS(I8,  I32, I8,  I8, KERNEL_SOURCE_4)
};

static const _kernel_map_type scatter_nd_update_special_copy_map[] =
{
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_COPY_KERNELS(U8,  I32, U8,  U8, KERNEL_SOURCE_4)
    TENSOR_SCATTER_ND_UPDATE_SPECIAL_COPY_KERNELS(I8,  I32, I8,  I8, KERNEL_SOURCE_4)
};

/*
 * Kernel params
 */
static vx_param_description_t _scatter_nd_update_reset_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_update_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_ref_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_copy_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_special_ref_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_special_update_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

static vx_param_description_t _scatter_nd_update_special_copy_kernel_param_def[] =
{
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED},
    // Add kererl parameters here
};

#define _SCATTER_ND_UPDATE_RESET_PARAM_NUM  _cnt_of_array( _scatter_nd_update_reset_kernel_param_def )
#define _SCATTER_ND_UPDATE_UPDATE_PARAM_NUM  _cnt_of_array(_scatter_nd_update_update_kernel_param_def)
#define _SCATTER_ND_UPDATE_REF_PARAM_NUM  _cnt_of_array(_scatter_nd_update_ref_kernel_param_def)
#define _SCATTER_ND_UPDATE_COPY_PARAM_NUM  _cnt_of_array(_scatter_nd_update_copy_kernel_param_def)

#define _SCATTER_ND_UPDATE_SPECIAL_REF_PARAM_NUM  _cnt_of_array(_scatter_nd_update_special_ref_kernel_param_def)
#define _SCATTER_ND_UPDATE_SPECIAL_UPDATE_PARAM_NUM  _cnt_of_array(_scatter_nd_update_special_update_kernel_param_def)
#define _SCATTER_ND_UPDATE_SPECIAL_COPY_PARAM_NUM  _cnt_of_array(_scatter_nd_update_special_copy_kernel_param_def)

static vsi_status get_scatter_nd_update_tensor_reshape_size
    (
    vsi_nn_tensor_t ** inputs,
    vsi_size_t sizes[VSI_NN_MAX_DIM_NUM],
    uint32_t block_size,
    uint32_t coordDim,
    vsi_size_t strides[VSI_NN_MAX_DIM_NUM],
    int32_t* newDim,
    int32_t* isBig
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t dims_num = inputs[0]->attr.dim_num;
    vsi_size_t *input_size = inputs[0]->attr.size;
    uint32_t i = 0;
    vsi_size_t elementCnt = 1;

#define VSI_NN_MAX_IMAGE_WIDTH  GPU_TENSOR_MAX_WIDTH

    newDim[0] = 0;
    for (i = 0; i < dims_num; ++i)
    {
        elementCnt *= input_size[i];
    }

    for (i = 0; i < VSI_NN_MAX_DIM_NUM; ++i)
    {
        sizes[i] = 1;
    }

    sizes[0] = block_size;
    sizes[1] = elementCnt / block_size;
    newDim[0] = 2;

    if ((elementCnt / block_size) >= VSI_NN_MAX_IMAGE_WIDTH)
    {
        isBig[0] |= 1;
    }

    if (coordDim == 1 && strides) // index shape
    {
        for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
        {
            strides[i] = 0;
        }
    }
    else if (coordDim >= 2 && coordDim <= VSI_NN_MAX_DIM_NUM && strides)
    {
        for (i = 0; i < VSI_NN_MAX_DIM_NUM; i++)
        {
            strides[i] = 0;
        }

        strides[0] = input_size[dims_num - coordDim];
        for (i = 1; i < coordDim - 1; i++)
        {
            strides[i] = strides[i - 1] * input_size[dims_num - coordDim + i];
        }
    }

#undef VSI_NN_MAX_IMAGE_WIDTH

    return status;
} /* _get_EltOP_tensor_reshape_size */

static vsi_status check_scatter_nd_update_index_repeat
    (
    vsi_nn_tensor_t ** inputs,
    int32_t coord_dim,
    int32_t block_size,
    int32_t indices_num,
    int32_t* isRepeat
    )
{
    vsi_status status = VSI_FAILURE;
    int32_t i = 0, j = 0;
    vsi_size_t elementNum = 1;
    vsi_nn_kernel_tensor_t ref_tensor = (vsi_nn_kernel_tensor_t)inputs[0]->t;
    vsi_nn_kernel_tensor_t index_tensor = (vsi_nn_kernel_tensor_t)inputs[1]->t;
    vsi_nn_kernel_tensor_attr_t* attr[2] = { NULL };
    uint32_t*   index_buffer[1] = { NULL };
    int32_t* mask_buffer = NULL;
    int32_t  mask_len = 0;

    if (indices_num == 1)
    {
        isRepeat[0] = 0;
        return VSI_SUCCESS;
    }

    if (inputs[1]->attr.is_const == FALSE)
    {
        isRepeat[0] = 1;
        return VSI_SUCCESS;
    }

    attr[0] = vsi_nn_kernel_tensor_attr_create( ref_tensor );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", final );

    attr[1] = vsi_nn_kernel_tensor_attr_create( index_tensor );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", final );

    elementNum = vsi_nn_kernel_tensor_attr_get_size( attr[0] );
    mask_len = (int32_t)elementNum / block_size;
    mask_buffer = (int32_t*)malloc(mask_len * sizeof(int32_t));
    CHECK_PTR_FAIL_GOTO( mask_buffer, "Create mask buffer fail.", final );
    memset(mask_buffer, 0, mask_len * sizeof(int32_t));

    index_buffer[0] = (uint32_t*)vsi_nn_kernel_tensor_create_buffer( index_tensor, attr[1], FALSE );
    CHECK_PTR_FAIL_GOTO( index_buffer[0], "Create index buffer fail.", final );

    if (coord_dim <= 5)
    {
        vsi_ssize_t stride[5] = {0, 0, 0, 0, 0};
        vsi_ssize_t new_shape[5] = {1, 1, 1, 1, 1};
        vsi_ssize_t merge_dim = (vsi_ssize_t)attr[0]->shape->size - coord_dim + 1;

        for (i = 0; i < (int32_t)merge_dim; ++i)
        {
            new_shape[0] *= attr[0]->shape->data[i];
        }
        stride[0] = new_shape[0] / block_size;

        for (i = 1; i < coord_dim; ++i)
        {
            new_shape[i] = attr[0]->shape->data[merge_dim + i - 1];

            stride[i] = stride[i - 1] * new_shape[i];
        }

        for (i = 0; i < indices_num; i++)
        {
            uint32_t coord[5] = {0};
            int32_t byd_flg = 0;
            vsi_ssize_t  mask_idx = 0;

            for (j = 0; j < coord_dim; j++)
            {
                coord[j] = index_buffer[0][i * coord_dim + coord_dim - j - 1];
                if (coord[j] >= (uint32_t)new_shape[j])
                {
                    byd_flg = 1;
                    break;
                }
            }
            if (byd_flg)
            {
                continue;
            }

            mask_idx = coord[4] * stride[3] + coord[3] * stride[2] +
                            coord[2] * stride[1] + coord[1] * stride[0] + coord[0];
            if (mask_buffer[mask_idx] == 0)
            {
                mask_buffer[mask_idx] = 1;
            }
            else if (mask_buffer[mask_idx] > 0)
            {
                isRepeat[0] = 1;
                status = VSI_SUCCESS;
                CHECK_STATUS_FAIL_GOTO( status, final );
            }
        }
    }
    else
    {
        status = VSI_FAILURE;
        CHECK_STATUS_FAIL_GOTO( status, final );
    }

final:
    if ( index_buffer[0] )
    {
        free( index_buffer[0] );
    }

    if ( mask_buffer )
    {
        free( mask_buffer );
    }

    for ( i = 0; i < 2; i ++ )
    {
        if (attr[i]) { vsi_nn_kernel_tensor_attr_release( &attr[i] ); }
    }

    return VSI_SUCCESS;
} /* check_scatter_nd_update_index_repeat */

/*
 * Kernel initializer
 */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_special_ref_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    int32_t block_size = 1;
    int32_t width = 0;
    int32_t height = 0;

    int32_t input0_zp    = 0;
    float   input0_scale = 1.0f;
    int32_t output_zp    = 0;
    float   output_scale = 1.0f;

    uint32_t pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    block_size   = (int32_t)(attr[0]->shape->data[0]);
    height = (int32_t)(attr[0]->shape->data[1]);
    width = (int32_t)(block_size * height);
    if (attr[0]->dtype == F16 || attr[0]->dtype == I16 || attr[0]->dtype == U16)
    {
        width = (width + 7) / 8;
    }
    else if (attr[0]->dtype == U8 || attr[0]->dtype == I8)
    {
        width = (width + 15) / 16;
    }

    input0_zp     = attr[0]->zero_point;
    input0_scale  = attr[0]->scale;
    output_zp     = attr[1]->zero_point;
    output_scale  = 1.0f / attr[1]->scale;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = width;
    gpu_param.global_size[1]   = 1;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[0]->dtype, attr[1]->dtype );

    switch( pack_key )
    {
    case _PACK_SELECT_KEY( I8,  I8 ):
    case _PACK_SELECT_KEY( U8,  U8 ):
        {
            uint16_t M0               = 0;
            int32_t  postShift0       = 0;
            uint32_t multAndoutZP0[2] = {0};

            gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniU8MulAndPostShift_Hi_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x1b1a1918, 0x1f1e1d1c, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            gpu_quantize_multiplier_16bit( (double)input0_scale * output_scale, &M0, &postShift0);

            multAndoutZP0[0] = (uint32_t)(M0);
            multAndoutZP0[1] = (uint32_t)((output_zp << postShift0) - input0_zp * M0);

            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift0 );
            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Hi_2x8, postShift0 );

            status = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift0_Lo_2x8",  &uniU8MulAndPostShift_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift0_Hi_2x8",  &uniU8MulAndPostShift_Hi_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
        break;
    default:
        break;
    }

#undef _PACK_SELECT_KEY

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    return status;
} /* _scatter_nd_update_special_ref_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_special_update_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t     block_size = 1;
    int32_t     update_width = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0, area = 0, vol = 0;
    int32_t     coord_dim  = 0;
    int32_t     offsetX = 0, offsetY = 0, offsetZ = 0, offsetW = 0, offset_idx = 0;
    int32_t     input1_zp    = 0;
    float       input1_scale = 1.0f;
    int32_t     output_zp    = 0;
    float       output_scale = 1.0f;
    uint32_t    pack_key = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &width);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &area);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &vol);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size   = (int32_t)(attr[2]->shape->data[0]);
    update_width = (int32_t)(attr[1]->shape->data[0]);
    index_num    = (int32_t)(attr[0]->shape->data[1]);

    input1_zp     = attr[1]->zero_point;
    input1_scale  = attr[1]->scale;
    output_zp     = attr[2]->zero_point;
    output_scale  = 1.0f / attr[2]->scale;

    if (coord_dim == 5)
    {
        offset_idx = 1;
    }
    if (coord_dim == 4 || coord_dim == 5)
    {
        offsetX = vol;
        offsetY = area;
        offsetZ = width;
        offsetW = 1;
    }
    else if (coord_dim == 3)
    {
        offsetX = area;
        offsetY = width;
        offsetZ = 1;
    }
    else if (coord_dim == 2)
    {
        offsetX = width;
        offsetY = 1;
        offsetZ = 0;
    }
    else if (coord_dim == 1)
    {
        offsetX = 1;
        offsetY = 0;
        offsetZ = 0;
    }

    if (attr[1]->dtype == F16 || attr[1]->dtype == I16 || attr[1]->dtype == U16)
    {
        update_width = (update_width + 7) / 8;
    }
    else if (attr[1]->dtype == U8 || attr[1]->dtype == I8)
    {
        update_width = (update_width + 15) / 16;
    }

    if (attr[2]->dtype == F16 || attr[2]->dtype == I16 || attr[2]->dtype == U16)
    {
        block_size = (block_size + 7) / 8;
    }
    else if (attr[2]->dtype == U8 || attr[2]->dtype == I8)
    {
        block_size = (block_size + 15) / 16;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = block_size;
    gpu_param.global_size[1]   = index_num;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        status = vsi_nn_kernel_gpu_add_param( node, "update_width", &update_width );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetX", &offsetX );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetY", &offsetY );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetZ", &offsetZ );
        status |= vsi_nn_kernel_gpu_add_param( node, "offsetW", &offsetW );
        status |= vsi_nn_kernel_gpu_add_param( node, "offset_idx", &offset_idx );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
    }
#define _PACK_SELECT_KEY( IN0_TYPE, OUT_TYPE )    \
        (IN0_TYPE | ( OUT_TYPE << 16))

    pack_key = _PACK_SELECT_KEY( attr[1]->dtype, attr[2]->dtype );

    switch( pack_key )
    {
    case _PACK_SELECT_KEY( I8,  I8 ):
    case _PACK_SELECT_KEY( U8,  U8 ):
        {
            uint16_t M1               = 0;
            int32_t  postShift1       = 0;
            uint32_t multAndoutZP1[2] = {0};

            gpu_dp_inst_t uniU8MulAndPostShift_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };
            gpu_dp_inst_t uniU8MulAndPostShift_Hi_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x1b1a1918, 0x1f1e1d1c, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

            gpu_quantize_multiplier_16bit( (double)input1_scale * output_scale, &M1, &postShift1);

            multAndoutZP1[0] = (uint32_t)(M1);
            multAndoutZP1[1] = (uint32_t)((output_zp << postShift1) - input1_zp * M1);

            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Lo_2x8, postShift1 );
            gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_Hi_2x8, postShift1 );

            status = vsi_nn_kernel_gpu_add_param( node, "multAndoutZP1", &multAndoutZP1 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift1_Lo_2x8",  &uniU8MulAndPostShift_Lo_2x8 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                "uniU8MulAndPostShift1_Hi_2x8",  &uniU8MulAndPostShift_Hi_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
        break;
    default:
        break;
    }
#undef _PACK_SELECT_KEY

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _scatter_nd_update_special_update_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_special_copy_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    int32_t     block_size = 1;
    int32_t     width = 0;
    int32_t     height = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    block_size   = (int32_t)(attr[0]->shape->data[0]);
    height = (int32_t)(attr[0]->shape->data[1]);
    width = (int32_t)(block_size * height);

    if (attr[0]->dtype == F16 || attr[0]->dtype == I16 || attr[0]->dtype == U16)
    {
        width = (width + 7) / 8;
    }
    else if (attr[0]->dtype == U8 || attr[0]->dtype == I8)
    {
        width = (width + 15) / 16;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = width;
    gpu_param.global_size[1]   = 1;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _scatter_nd_update_special_copy_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_reset_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        1,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[2] = { NULL };
    int32_t     width         = 0;
    int32_t     element_size  = 1;
    int32_t     input_zp0     = 0;
    float       input_scale0  = 1;
    int32_t     output_zp     = 0;
    float       output_scale  = 1;
    int32_t     i             = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );

    for (i = 0; i < (int32_t)attr[0]->shape->size; i++)
    {
        element_size *= (int32_t)attr[0]->shape->data[i];
    }
    width = element_size / 8;

    input_zp0     = attr[0]->zero_point;
    input_scale0  = attr[0]->scale;
    output_zp     = attr[1]->zero_point;
    output_scale  = attr[1]->scale;

    if (attr[0]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        input_scale0 = 1.0f;
    }
    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        output_scale = 1.0f;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    if (element_size < 8)
    {
        gpu_param.global_size[0]   = element_size;
    }
    else
    {
        gpu_param.global_size[0]   = width;
    }
    gpu_param.global_size[1]   = 1;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        uint16_t M0                = 0;
        int32_t  postShift0        = 0;
        uint32_t multAndoutZP0[2]  = {0};
        gpu_dp_inst_t uniU8MulAndPostShift_0_Lo_2x8 = {{
                0xdddddddd, // TCfg
                0x44444444, // ASelt
                0x13121110, 0x17161514, // ABin
                0x11111111, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002600, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_quantize_multiplier_16bit( (double)input_scale0 / output_scale, &M0, &postShift0);
        multAndoutZP0[0] = (uint32_t)(M0);
        multAndoutZP0[1] = (uint32_t)((output_zp << postShift0) - input_zp0 * M0);
        gpu_dp_inst_update_postshfit( &uniU8MulAndPostShift_0_Lo_2x8, postShift0 );

        status = vsi_nn_kernel_gpu_add_param( node,
                    "uniU8MulAndPostShift_0_Lo_2x8", &uniU8MulAndPostShift_0_Lo_2x8 );
        status |= vsi_nn_kernel_gpu_add_param( node, "multAndoutZP0", &multAndoutZP0 );
        CHECK_STATUS_FAIL_GOTO(status, OnError);
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    return status;
} /* _scatter_nd_update_reset_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_update_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        2,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t     block_size = 1;
    int32_t     update_width = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0;
    int32_t     coord_dim  = 0;
    int32_t     strides[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t     coord_strides[8]  = {0};
    int32_t     coord_strides1[4] = {0};
    int32_t     input2_zp = 0;
    int32_t     i = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[2] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[4], &strides[0]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[5], &strides[1]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &strides[2]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &strides[3]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &strides[4]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &strides[5]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &strides[6]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[11], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size   = (int32_t)(attr[2]->shape->data[0]);
    update_width = (int32_t)(attr[1]->shape->data[0]);
    index_num    = (int32_t)(attr[0]->shape->data[1]);
    width = block_size;
    if (block_size % 4 == 0)
    {
        update_width = update_width / 4;
        width = block_size / 4;
    }

    input2_zp     = attr[1]->zero_point;

    coord_strides[coord_dim - 1] = 1;
    for (i = 0; i < coord_dim - 1; i++)
    {
        coord_strides[i] = strides[coord_dim - 2 - i];
    }
    memcpy(coord_strides1, coord_strides + 4, 4 * sizeof(int32_t));

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = width;
    gpu_param.global_size[1]   = index_num;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvert1stUint8SubZpToFp32_4x4 = {{
                0x05050505, // TCfg
                0x04040404, // ASelt
                0x00010000, 0x00030002, // ABin
                0x0a0a0a0a, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00000400, // AccumType, ConstantType, and PostShift
                0xffff0001, 0x00000000, 0xffff0001, 0x00000000,
                0xffff0001, 0x00000000, 0xffff0001, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvertFp16ToFp32_4x4 = {{
            0x01010101, // TCfg
            0x00000000, // ASelt
            0x00010000, 0x00030002, // ABin
            0x02020202, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000100, // AccumType, ConstantType, and PostShift
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000,
            0x00003c00, 0x00000000, 0x00003c00, 0x00000000 // Constant
        }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniConvBF16toF32_Part0_2x8 = {{
            0x11111111, // TCfg
            0x01010101, // ASelt
            0x01050004, 0x03070206, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "update_width", &update_width );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_width", &block_size );
        status |= vsi_nn_kernel_gpu_add_param( node, "coord_stride", &coord_strides );
        status |= vsi_nn_kernel_gpu_add_param( node, "coord_stride1", &coord_strides1 );
        CHECK_STATUS_FAIL_GOTO(status, OnError);

        if (attr[1]->dtype == U8 || attr[1]->dtype == I8 || attr[1]->dtype == I16)
        {
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvert1stUint8SubZpToFp32_4x4",  &uniConvert1stUint8SubZpToFp32_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node, "input_zp", &input2_zp );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
        else if (attr[1]->dtype == F16 || attr[1]->dtype == BF16)
        {
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertFp16ToFp32_4x4",  &uniConvertFp16ToFp32_4x4 );
            status |= vsi_nn_kernel_gpu_add_param( node,
                    "uniConvBF16toF32_Part0_2x8",  &uniConvBF16toF32_Part0_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, OnError );
        }
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _scatter_nd_update_update_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_ref_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        3,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[3] = { NULL };
    int32_t     block_size = 1;
    int32_t     update_width = 1;
    int32_t     index_num  = 1;
    int32_t     width = 0;
    int32_t     coord_dim  = 0;
    int32_t     strides[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t     coord_strides[8]  = {0};
    int32_t     coord_strides1[4] = {0};
    float       output_zp = 0;
    float       input_scale = 1.0f;
    float       output_scale = 1.0f;
    float       inout_scale = 1.0f;
    int32_t     i = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );
    attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );
    CHECK_PTR_FAIL_GOTO( attr[1], "Create tensor attr buffer fail.", OnError );
    attr[2] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[3] );
    CHECK_PTR_FAIL_GOTO( attr[2], "Create tensor attr buffer fail.", OnError );

    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[6], &strides[0]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[7], &strides[1]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[8], &strides[2]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[9], &strides[3]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[10], &strides[4]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[11], &strides[5]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[12], &strides[6]);
    CHECK_STATUS_FAIL_GOTO(status, OnError );
    status = vsi_nn_kernel_scalar_read_int32((vsi_nn_kernel_scalar_t)param[13], &coord_dim);
    CHECK_STATUS_FAIL_GOTO(status, OnError );

    block_size   = (int32_t)(attr[2]->shape->data[0]);
    update_width = (int32_t)(attr[1]->shape->data[0]);
    index_num    = (int32_t)(attr[0]->shape->data[1]);

    input_scale  = attr[1]->scale;
    output_scale = attr[2]->scale;
    output_zp    = (float)attr[2]->zero_point;
    if (attr[1]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        input_scale = 1.0f;
    }
    if (attr[2]->quant == VSI_NN_KERNEL_QUANT_NONE)
    {
        output_scale = 1.0f;
    }
    inout_scale   = input_scale / output_scale;

    coord_strides[coord_dim - 1] = 1;
    for (i = 0; i < coord_dim - 1; i++)
    {
        coord_strides[i] = strides[coord_dim - 2 - i];
    }
    memcpy(coord_strides1, coord_strides + 4, 4 * sizeof(int32_t));

    width = block_size;
    if (block_size % 4 == 0)
    {
        width = block_size / 4;
    }

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;

    gpu_param.global_size[0]   = width;
    gpu_param.global_size[1]   = index_num;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

    {
        gpu_dp_inst_t uniConvertInt32toUint8_2x8 = {{
                0x33333333, // TCfg
                0x11110000, // ASelt
                0x03020100, 0x03020100, // ABin
                0x00000000, // BSelt
                0x00000000, 0x00000000, // BBin
                0x00002400, // AccumType, ConstantType, and PostShift
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 // Constant
            }, GPU_DP_TYPE_16 };

        gpu_dp_inst_t uniExtractOddData_2x8 = {{
            0x11111111, // TCfg
            0x11110000, // ASelt
            0x07050301, 0x07050301, // ABin
            0x22222222, // BSelt
            0x00000000, 0x00000000, // BBin
            0x00000600, // AccumType, ConstantType, and PostShift
            0x00000001, 0x00000001, 0x00000001, 0x00000001,
            0x00000001, 0x00000001, 0x00000001, 0x00000001 // Constant
        }, GPU_DP_TYPE_16};

        status = vsi_nn_kernel_gpu_add_param( node, "output_stride", &width );
        status |= vsi_nn_kernel_gpu_add_param( node, "ref_stride", &update_width );
        status |= vsi_nn_kernel_gpu_add_param( node, "coord_stride", &coord_strides );
        status |= vsi_nn_kernel_gpu_add_param( node, "coord_stride1", &coord_strides1 );
        status |= vsi_nn_kernel_gpu_add_param( node, "output_zp", &output_zp );
        status |= vsi_nn_kernel_gpu_add_param( node, "inout_scale", &inout_scale );
        CHECK_STATUS_FAIL_GOTO(status, OnError);

        if (attr[1]->dtype == U8 || attr[1]->dtype == I8 || attr[1]->dtype == I16)
        {
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniConvertInt32toUint8_2x8",  &uniConvertInt32toUint8_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, OnError);
        }
        else if (attr[1]->dtype == BF16)
        {
            status = vsi_nn_kernel_gpu_add_param( node,
                    "uniExtractOddData_2x8",  &uniExtractOddData_2x8 );
            CHECK_STATUS_FAIL_GOTO(status, OnError);
        }
    }

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    if (attr[1])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[1] );
        attr[1] = NULL;
    }
    if (attr[2])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[2] );
        attr[2] = NULL;
    }
    return status;
} /* _scatter_nd_update_ref_initializer() */

DEF_KERNEL_INITIALIZER(_scatter_nd_update_copy_initializer)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    gpu_param_t gpu_param = {
        1,
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
        };

    vsi_nn_kernel_tensor_attr_t * attr[1] = { NULL };
    int32_t     width         = 0;
    int32_t     element_size  = 1;
    int32_t     i             = 0;

    VSI_UNREFERENCED(param_size);

    attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    CHECK_PTR_FAIL_GOTO( attr[0], "Create tensor attr buffer fail.", OnError );

    for (i = 0; i < (int32_t)attr[0]->shape->size; i++)
    {
        element_size *= (int32_t)attr[0]->shape->data[i];
    }
    width = element_size / 8;

    gpu_param.global_scale[0]  = 1;
    gpu_param.global_scale[1]  = 1;
    gpu_param.global_scale[2]  = 1;
    if (element_size < 8)
    {
        gpu_param.global_size[0]   = element_size;
    }
    else
    {
        gpu_param.global_size[0]   = width;
    }
    gpu_param.global_size[1]   = 1;
    gpu_param.global_size[2]   = 1;

    status = vsi_nn_kernel_gpu_config( node, &gpu_param );
    CHECK_STATUS_FAIL_GOTO(status, OnError);

OnError:
    if (attr[0])
    {
        vsi_nn_kernel_tensor_attr_release( &attr[0] );
        attr[0] = NULL;
    }
    return status;
} /* _scatter_nd_update_copy_initializer() */

static vsi_status _query_kernel
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_reset,
    vsi_nn_kernel_t* kernel_update,
    vsi_nn_kernel_t* kernel_ref,
    vsi_nn_kernel_t* kernel_copy,
    int32_t coord_flg,
    int32_t opt_flg
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    vsi_nn_kernel_dtype_e acc_dtype    = I32;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    if (input2_dtype == F16)
    {
        acc_dtype = F32;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, 0, output_dtype, 0, 0, 0);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_reset_map); i ++ )
    {
        if ( scatter_nd_update_reset_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(scatter_nd_update_reset_map) )
    {
        snprintf( kernel_reset->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_reset_map[i].function_name );
        kernel_reset->info.parameters = _scatter_nd_update_reset_kernel_param_def;
        kernel_reset->info.numParams = _SCATTER_ND_UPDATE_RESET_PARAM_NUM;
        kernel_reset->info.initialize = _scatter_nd_update_reset_initializer;

        vsi_nn_kernel_add_source( kernel_reset, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_reset_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_reset, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_reset_map[i].source_name );
    }
    else
    {
        status = VSI_FAILURE;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( 0, input2_dtype, 0, 1, coord_flg, opt_flg);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_update_map); i ++ )
    {
        if ( scatter_nd_update_update_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_update_map) )
    {
        snprintf( kernel_update->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_update_map[i].function_name );
        kernel_update->info.parameters = _scatter_nd_update_update_kernel_param_def;
        kernel_update->info.numParams = _SCATTER_ND_UPDATE_UPDATE_PARAM_NUM;
        kernel_update->info.initialize = _scatter_nd_update_update_initializer;

        vsi_nn_kernel_add_source( kernel_update, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_update_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_update, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_update_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( 0, acc_dtype, output_dtype, 2, coord_flg, opt_flg);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_ref_map); i ++ )
    {
        if ( scatter_nd_update_ref_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(scatter_nd_update_ref_map) )
    {
        snprintf( kernel_ref->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_ref_map[i].function_name );
        kernel_ref->info.parameters = _scatter_nd_update_ref_kernel_param_def;
        kernel_ref->info.numParams = _SCATTER_ND_UPDATE_REF_PARAM_NUM;
        kernel_ref->info.initialize = _scatter_nd_update_ref_initializer;

        vsi_nn_kernel_add_source( kernel_ref, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_ref_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_ref, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_ref_map[i].source_name );
    }
    else
    {
        status = VSI_FAILURE;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( 0, 0, output_dtype, 3, 0, 0);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_copy_map); i ++ )
    {
        if ( scatter_nd_update_copy_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_copy_map) )
    {
        snprintf( kernel_copy->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_copy_map[i].function_name );
        kernel_copy->info.parameters = _scatter_nd_update_copy_kernel_param_def;
        kernel_copy->info.numParams = _SCATTER_ND_UPDATE_COPY_PARAM_NUM;
        kernel_copy->info.initialize = _scatter_nd_update_copy_initializer;

        vsi_nn_kernel_add_source( kernel_copy, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_copy_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_copy, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_copy_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }

    return status;
} /* _query_kernel() */

static vsi_status _query_kernel_special
    (
    vsi_nn_tensor_t* const* const inputs,
    vsi_nn_tensor_t* const* const outputs,
    vsi_nn_kernel_t* kernel_ref,
    vsi_nn_kernel_t* kernel_update,
    vsi_nn_kernel_t* kernel
    )
{
    vsi_status status = VSI_SUCCESS;
    vsi_nn_kernel_dtype_e input0_dtype = U8;
    vsi_nn_kernel_dtype_e input2_dtype = F16;
    vsi_nn_kernel_dtype_e output_dtype = U8;
    uint32_t key = 0;
    size_t i = 0;

    input0_dtype = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    input2_dtype = vsi_nn_kernel_map_dtype( inputs[2]->attr.dtype.vx_type );
    output_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, input2_dtype, output_dtype, 4, 1, 0);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_special_ref_map); i ++ )
    {
        if ( scatter_nd_update_special_ref_map[i].key == key )
        {
            break;
        }
    }

    if ( i < _cnt_of_array(scatter_nd_update_special_ref_map) )
    {
        snprintf( kernel_ref->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_special_ref_map[i].function_name );
        kernel_ref->info.parameters = _scatter_nd_update_special_ref_kernel_param_def;
        kernel_ref->info.numParams = _SCATTER_ND_UPDATE_SPECIAL_REF_PARAM_NUM;
        kernel_ref->info.initialize = _scatter_nd_update_special_ref_initializer;

        vsi_nn_kernel_add_source( kernel_ref, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_special_ref_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_ref, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_special_ref_map[i].source_name );
    }
    else
    {
        status = VSI_FAILURE;
    }


    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, input2_dtype, output_dtype, 5, 1, 0);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_special_update_map); i ++ )
    {
        if ( scatter_nd_update_special_update_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_special_update_map) )
    {
        snprintf( kernel_update->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_special_update_map[i].function_name );
        kernel_update->info.parameters = _scatter_nd_update_special_update_kernel_param_def;
        kernel_update->info.numParams = _SCATTER_ND_UPDATE_SPECIAL_UPDATE_PARAM_NUM;
        kernel_update->info.initialize = _scatter_nd_update_special_update_initializer;

        vsi_nn_kernel_add_source( kernel_update, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_special_update_map[i].source_name );
        vsi_nn_kernel_add_source( kernel_update, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_special_update_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }

    key = HASH_SCATTER_ND_UPDATE_KEY( input0_dtype, input2_dtype, output_dtype, 6, 1, 0);

    for ( i = 0; i < _cnt_of_array(scatter_nd_update_special_copy_map); i ++ )
    {
        if ( scatter_nd_update_special_copy_map[i].key == key )
        {
            break;
        }
    }
    if ( i < _cnt_of_array(scatter_nd_update_special_copy_map) )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",
                        scatter_nd_update_special_copy_map[i].function_name );
        kernel->info.parameters = _scatter_nd_update_special_copy_kernel_param_def;
        kernel->info.numParams = _SCATTER_ND_UPDATE_SPECIAL_COPY_PARAM_NUM;
        kernel->info.initialize = _scatter_nd_update_special_copy_initializer;

        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                scatter_nd_update_special_copy_map[i].source_name );
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                scatter_nd_update_special_copy_map[i].source_name );
    }
    else
    {
        status |= VSI_FAILURE;
    }
    return status;
} /* _query_kernel_special() */

static vsi_nn_kernel_node_t _setup
    (
    vsi_nn_graph_t              * graph,
    vsi_nn_tensor_t            ** inputs,
    size_t                        input_num,
    vsi_nn_tensor_t            ** outputs,
    size_t                        output_num,
    const vsi_nn_kernel_param_t * params,
    vsi_nn_kernel_t             * kernel
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_kernel_node_t node = NULL;
    vsi_size_t  shapes[3][VSI_NN_MAX_DIM_NUM] = {{0}};
    vsi_size_t  strides[VSI_NN_MAX_DIM_NUM] = {0};
    int32_t block_size  = vsi_nn_kernel_param_get_int32( params, "block_size" );
    int32_t coord_dim   = vsi_nn_kernel_param_get_int32( params, "coord_dim" );
    int32_t idx_num  = vsi_nn_kernel_param_get_int32( params, "idx_num" );
    int32_t rs_in_dim = 0, rs_idx_dim = 0, rs_out_dim = 0;
    int32_t big_flg = 0;
    vsi_nn_kernel_dtype_e update_dtype = vsi_nn_kernel_map_dtype(inputs[2]->attr.dtype.vx_type);
    vsi_nn_kernel_dtype_e ref_dtype = vsi_nn_kernel_map_dtype(inputs[0]->attr.dtype.vx_type);
    vsi_nn_kernel_dtype_e output_dtype = vsi_nn_kernel_map_dtype(outputs[0]->attr.dtype.vx_type);
    int32_t type_flg = ((update_dtype == U8 || update_dtype == I8 || update_dtype == I16) &&
                        (update_dtype == ref_dtype && update_dtype == output_dtype)) ? 1 : 0;
    int32_t special_flg = (block_size % 16 == 0 && type_flg && coord_dim <= 4)  ? 1 : 0;
    int32_t coord_flg = 0;
    int32_t opt_flg = (block_size % 4 == 0) ? 1 : 0;
    int32_t i = 0;
    int32_t isRepeat = 0;
    vsi_nn_tensor_t * tensors[4] = { NULL };
    vsi_nn_kernel_t * ikernels[3] = { NULL };

    VSI_UNREFERENCED(input_num);
    VSI_UNREFERENCED(output_num);

    status = get_scatter_nd_update_tensor_reshape_size(&inputs[1], shapes[0], coord_dim, 0,
                                                    NULL, &rs_idx_dim, &big_flg);
    status |= get_scatter_nd_update_tensor_reshape_size(&inputs[2], shapes[1], block_size, 0,
                                                    NULL, &rs_in_dim, &big_flg);
    status |= get_scatter_nd_update_tensor_reshape_size(&outputs[0], shapes[2], block_size, coord_dim,
                                                    strides, &rs_out_dim, &big_flg);
    CHECK_STATUS_FAIL_GOTO( status, final );

    check_scatter_nd_update_index_repeat(inputs, coord_dim, block_size, idx_num, &isRepeat);

    if (special_flg && isRepeat == 0)
    {
        vsi_nn_tensor_attr_t attr;
        vsi_nn_kernel_node_t tmp_node = NULL;
        vsi_nn_kernel_node_t ref_node = NULL;
        vsi_nn_kernel_node_param_t ref_params[_SCATTER_ND_UPDATE_SPECIAL_REF_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t node_params[_SCATTER_ND_UPDATE_SPECIAL_UPDATE_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t cpy_params[_SCATTER_ND_UPDATE_SPECIAL_COPY_PARAM_NUM] = { NULL };

        ikernels[0] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[0]->unique_id = kernel->unique_id;
        ikernels[1] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[1]->unique_id = kernel->unique_id;

        memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        attr.dtype = outputs[0]->attr.dtype;
        attr.is_const = FALSE;
        attr.vtl = TRUE;

        for (i = 0; i < rs_out_dim; i++)
        {
            attr.size[i] = shapes[2][i];
        }
        attr.dim_num = rs_out_dim;

        tensors[0] = vsi_nn_CreateTensor( graph, &attr );
        attr.size[0] = 1;
        attr.size[1] = 1;
        tensors[1] = vsi_nn_CreateTensor( graph, &attr );
        tensors[2] = vsi_nn_CreateTensor( graph, &attr );

        status = _query_kernel_special( inputs, outputs, ikernels[0], ikernels[1], kernel);
        if ( VSI_SUCCESS == status)
        {
            // convert ref to output
            ref_node = vsi_nn_kernel_create_node( graph, ikernels[0] );
            if (ref_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                ref_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[2], rs_out_dim );
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                status = vsi_nn_kernel_node_pass_param( ref_node, ref_params,
                                _SCATTER_ND_UPDATE_SPECIAL_REF_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &ref_params[0] );
            }

            // update
            tmp_node = vsi_nn_kernel_create_node( graph, ikernels[1] );
            if (tmp_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                node_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                node_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[0] );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[1] );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[2] );
                node_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( tmp_node, node_params,
                                _SCATTER_ND_UPDATE_SPECIAL_UPDATE_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &node_params[0] );
                vsi_nn_kernel_tensor_release( &node_params[1] );
                vsi_nn_kernel_scalar_release( &node_params[5] );
                vsi_nn_kernel_scalar_release( &node_params[6] );
                vsi_nn_kernel_scalar_release( &node_params[7] );
                vsi_nn_kernel_scalar_release( &node_params[8] );
            }

            // copy to output
            node = vsi_nn_kernel_create_node( graph, kernel );
            if ( node )
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                cpy_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                cpy_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                cpy_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
                status = vsi_nn_kernel_node_pass_param( node, cpy_params, _SCATTER_ND_UPDATE_SPECIAL_COPY_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &cpy_params[2] );
            }
        }

        if ( ikernels[0] )
        {
            vsi_nn_kernel_release( &ikernels[0] );
        }
        if ( ikernels[1] )
        {
            vsi_nn_kernel_release( &ikernels[1] );
        }
        if ( tensors[0] )
        {
            vsi_nn_ReleaseTensor( &tensors[0] );
        }
        if ( tensors[1] )
        {
            vsi_nn_ReleaseTensor( &tensors[1] );
        }
        if ( tensors[2] )
        {
            vsi_nn_ReleaseTensor( &tensors[2] );
        }
        if (ref_node) {vsi_nn_kernel_node_release( &ref_node );}
        if (tmp_node) {vsi_nn_kernel_node_release( &tmp_node );}
    }
    else
    {
        vsi_nn_tensor_attr_t attr;
        vsi_nn_kernel_node_t reset_node = NULL;
        vsi_nn_kernel_node_t update_node = NULL;
        vsi_nn_kernel_node_t ref_node = NULL;
        vsi_nn_kernel_node_param_t reset_params[_SCATTER_ND_UPDATE_RESET_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t ref_params[_SCATTER_ND_UPDATE_REF_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t update_params[_SCATTER_ND_UPDATE_UPDATE_PARAM_NUM] = { NULL };
        vsi_nn_kernel_node_param_t cpy_params[_SCATTER_ND_UPDATE_COPY_PARAM_NUM] = { NULL };
        int32_t width = 1;
        int32_t res = 0;

        ikernels[0] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[0]->unique_id = kernel->unique_id;
        ikernels[1] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[1]->unique_id = kernel->unique_id;
        ikernels[2] = vsi_nn_kernel_create( VSI_NN_KERNEL_TYPE_EVIS );
        ikernels[2]->unique_id = kernel->unique_id;

        memset( &attr, 0, sizeof(vsi_nn_tensor_attr_t) );
        attr.dtype = outputs[0]->attr.dtype;
        attr.is_const = FALSE;
        attr.vtl = TRUE;

        for (i = 0; i < rs_out_dim; i++)
        {
            attr.size[i] = shapes[2][i];
            width *= (int32_t)shapes[2][i];
        }
        attr.dim_num = rs_out_dim;

        res = width % 8;
        width = (width >> 3) << 3;

        tensors[0] = vsi_nn_CreateTensor( graph, &attr );  // ref'
        attr.dtype = inputs[2]->attr.dtype;
        attr.dtype.vx_type = VSI_NN_TYPE_INT32;
        attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        if (update_dtype == F16)
        {
            attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        }
        tensors[1] = vsi_nn_CreateTensor( graph, &attr );  // temp_buf_int
        attr.size[0] = 1;
        attr.size[1] = 1;
        tensors[2] = vsi_nn_CreateTensor( graph, &attr );  // link_buffer0
        tensors[3] = vsi_nn_CreateTensor( graph, &attr );  // link_buffer1

        status = _query_kernel( inputs, outputs, ikernels[0], ikernels[1], ikernels[2], kernel, coord_flg, opt_flg);
        if ( VSI_SUCCESS == status)
        {
            // convert ref to output
            reset_node = vsi_nn_kernel_create_node( graph, ikernels[0] );
            if (reset_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                reset_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[0]->t,  shapes[2], rs_out_dim );
                reset_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                reset_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                reset_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
                reset_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &res );
                status = vsi_nn_kernel_node_pass_param( reset_node, reset_params, _SCATTER_ND_UPDATE_RESET_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &reset_params[0] );
                vsi_nn_kernel_scalar_release( &reset_params[3] );
                vsi_nn_kernel_scalar_release( &reset_params[4] );
            }

            // update
            update_node = vsi_nn_kernel_create_node( graph, ikernels[1] );
            if (update_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                update_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                update_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                update_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                update_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[0] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[1] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[2] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[3] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[4] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[5] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[6] );
                update_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( update_node, update_params,
                                _SCATTER_ND_UPDATE_UPDATE_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &update_params[0] );
                vsi_nn_kernel_tensor_release( &update_params[1] );
                vsi_nn_kernel_scalar_release( &update_params[4] );
                vsi_nn_kernel_scalar_release( &update_params[5] );
                vsi_nn_kernel_scalar_release( &update_params[6] );
                vsi_nn_kernel_scalar_release( &update_params[7] );
                vsi_nn_kernel_scalar_release( &update_params[8] );
                vsi_nn_kernel_scalar_release( &update_params[9] );
                vsi_nn_kernel_scalar_release( &update_params[10] );
                vsi_nn_kernel_scalar_release( &update_params[11] );
            }

            // ref
            ref_node = vsi_nn_kernel_create_node( graph, ikernels[2] );
            if (ref_node)
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                ref_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[1]->t,  shapes[0], rs_idx_dim );
                ref_params[index++] = vsi_nn_kernel_tensor_reshape( inputs[2]->t,  shapes[1], rs_in_dim );
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[1]->t;
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[2]->t;
                ref_params[index++] = (vsi_nn_kernel_node_param_t)tensors[3]->t;
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[0] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[1] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[2] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[3] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[4] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[5] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &strides[6] );
                ref_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &coord_dim );
                status = vsi_nn_kernel_node_pass_param( ref_node, ref_params, _SCATTER_ND_UPDATE_REF_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &ref_params[0] );
                vsi_nn_kernel_tensor_release( &ref_params[1] );
                vsi_nn_kernel_scalar_release( &ref_params[6] );
                vsi_nn_kernel_scalar_release( &ref_params[7] );
                vsi_nn_kernel_scalar_release( &ref_params[8] );
                vsi_nn_kernel_scalar_release( &ref_params[9] );
                vsi_nn_kernel_scalar_release( &ref_params[10] );
                vsi_nn_kernel_scalar_release( &ref_params[11] );
                vsi_nn_kernel_scalar_release( &ref_params[12] );
                vsi_nn_kernel_scalar_release( &ref_params[13] );
            }

            // copy to output
            node = vsi_nn_kernel_create_node( graph, kernel );
            if ( node )
            {
                uint32_t index = 0;
                /* Pass parameters to node. */
                cpy_params[index++] = (vsi_nn_kernel_node_param_t)tensors[0]->t;
                cpy_params[index++] = (vsi_nn_kernel_node_param_t)tensors[3]->t;
                cpy_params[index++] = vsi_nn_kernel_tensor_reshape( outputs[0]->t, shapes[2], rs_out_dim );
                cpy_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &width );
                cpy_params[index++] = vsi_nn_kernel_scalar_create( graph, I32, &res );
                status = vsi_nn_kernel_node_pass_param( node, cpy_params, _SCATTER_ND_UPDATE_COPY_PARAM_NUM );
                CHECK_STATUS(status);
                vsi_nn_kernel_tensor_release( &cpy_params[2] );
                vsi_nn_kernel_scalar_release( &cpy_params[3] );
                vsi_nn_kernel_scalar_release( &cpy_params[4] );
            }
        }

        if ( ikernels[0] )
        {
            vsi_nn_kernel_release( &ikernels[0] );
        }
        if ( ikernels[1] )
        {
            vsi_nn_kernel_release( &ikernels[1] );
        }
        if ( ikernels[2] )
        {
            vsi_nn_kernel_release( &ikernels[2] );
        }
        if ( tensors[0] )
        {
            vsi_nn_ReleaseTensor( &tensors[0] );
        }
        if ( tensors[1] )
        {
            vsi_nn_ReleaseTensor( &tensors[1] );
        }
        if ( tensors[2] )
        {
            vsi_nn_ReleaseTensor( &tensors[2] );
        }
        if ( tensors[3] )
        {
            vsi_nn_ReleaseTensor( &tensors[3] );
        }
        if (ref_node) {vsi_nn_kernel_node_release( &ref_node );}
        if (reset_node) {vsi_nn_kernel_node_release( &reset_node );}
        if (update_node) {vsi_nn_kernel_node_release( &update_node );}
    }

final:
    if (ikernels[0])
    {
        vsi_nn_kernel_release(&ikernels[0]);
    }
    if (ikernels[1])
    {
        vsi_nn_kernel_release(&ikernels[1]);
    }
    if (ikernels[2])
    {
        vsi_nn_kernel_release(&ikernels[2]);
    }
    vsi_safe_release_tensor(tensors[0]);
    vsi_safe_release_tensor(tensors[1]);
    vsi_safe_release_tensor(tensors[2]);
    vsi_safe_release_tensor(tensors[3]);

    return node;
} /* _setup() */

__END_DECLS

REGISTER_BACKEND_EVIS( scatter_nd_update, _setup )
