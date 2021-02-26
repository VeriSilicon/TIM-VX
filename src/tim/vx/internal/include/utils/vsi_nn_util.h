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
/** @file */
#ifndef _VSI_NN_UTIL_H
#define _VSI_NN_UTIL_H

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "vsi_nn_platform.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"
#include "vsi_nn_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

#define vsi_nn_safe_free( _PTR ) if( _PTR ){ \
    free( _PTR ); _PTR = NULL; }

#define vsi_safe_release_tensor(_t) if(_t){vsi_nn_ReleaseTensor(&(_t)); _t = NULL;}

#define END_OF_VARIADIC_ARGUMENTS       0xbadcaffe
#define FOREACH_ARGS(_args, _next, _arg_type) \
    while(((_arg_type)((size_t)END_OF_VARIADIC_ARGUMENTS)) != (_next = va_arg(_args, _arg_type)))

/*-------------------------------------------
                  Functions
-------------------------------------------*/

/**
 * Load binary data from file
 * Load binary data from file, it will malloc the buffer to store
 * the data, user need to free it with vsi_nn_Free().
 * @see vsi_nn_Free
 *
 * @param[in] filename Binary data file path.
 * @param[out] sz Size(bytes) of data.
 *
 * @return Data buffer on success, or NULL otherwise.
 */
OVXLIB_API uint8_t * vsi_nn_LoadBinaryData
    (
    const char * filename,
    uint32_t  * sz
    );

OVXLIB_API uint32_t vsi_nn_GetStrideSize
    (
    vsi_nn_tensor_attr_t * attr,
    uint32_t            * stride
    );

OVXLIB_API uint32_t vsi_nn_GetStrideSizeBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
    vsi_nn_type_e type,
    uint32_t   * stride
    );

OVXLIB_API uint32_t vsi_nn_GetTotalBytesBySize
    (
    uint32_t   * size,
    uint32_t     dim_num,
    vsi_nn_type_e type
    );

/**
 * Convert data to float32
 * Convert data from any type to float32.
 *
 * @param[in] data The scalar data address.
 * @param[in] type Data type.
 *
 * @return Converted float32 data.
 */
OVXLIB_API float vsi_nn_DataAsFloat32
    (
    uint8_t    * data,
    vsi_nn_type_e type
    );

OVXLIB_API void vsi_nn_UpdateTensorDims
    (
    vsi_nn_tensor_attr_t * attr
    );

OVXLIB_API uint32_t vsi_nn_ComputeFilterSize
    (
    uint32_t   i_size,
    uint32_t   ksize,
    uint32_t * pad,
    uint32_t   stride,
    uint32_t   dilation,
    vsi_nn_round_type_e rounding
    );

OVXLIB_API void vsi_nn_InitTensorsId
    (
    vsi_nn_tensor_id_t * ids,
    int                  num
    );

OVXLIB_API void vsi_nn_ComputePadWithPadType
    (
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    );

OVXLIB_API void vsi_nn_ComputePadWithPadTypeForConv1D
    (
    uint32_t   * in_shape,
    uint32_t     in_dim_num,
    uint32_t   * ksize,
    uint32_t   * stride,
    vsi_nn_pad_e pad_type,
    vsi_nn_round_type_e rounding,
    uint32_t   * out_pad
    );

OVXLIB_API void vsi_nn_GetPadForOvx
    (
    uint32_t * in_pad,
    uint32_t * out_pad
    );

OVXLIB_API vsi_bool vsi_nn_CreateTensorGroup
    (
    vsi_nn_graph_t  *  graph,
    vsi_nn_tensor_t *  in_tensor,
    uint32_t          axis,
    vsi_nn_tensor_t ** out_tensors,
    uint32_t          group_number
    );

OVXLIB_API uint32_t vsi_nn_ShapeToString
    (
    uint32_t * shape,
    uint32_t   dim_num,
    char      * buf,
    uint32_t   buf_sz,
    vsi_bool     for_print
    );

OVXLIB_API int32_t vsi_nn_Access
    (
    const char *path,
    int32_t mode
    );

OVXLIB_API int32_t vsi_nn_Mkdir
    (
    const char *path,
    int32_t mode
    );

OVXLIB_API vsi_bool vsi_nn_CheckFilePath
    (
    const char *path
    );

/**
 * Malloc aligned buffer
 * Malloc address and size aligned buffer.
 *
 * @param[in] mem_size Buffer size to malloc.
 * @param[in] align_start_size Address aligned bytes.
 * @param[in] align_block_size Buffer size aligned bytes.
 *
 * @return The aligned buffer address on success, or NULL otherwise.
 */
OVXLIB_API uint8_t * vsi_nn_MallocAlignedBuffer
    (
    uint32_t mem_size,
    uint32_t align_start_size,
    uint32_t align_block_size
    );

/**
 * Free aligned buffer
 * Free aligend buffer malloc with vsi_nn_MallocAlignedBuffer().
 *
 * @param[in] handle Buffer handle to free.
 * @see vsi_nn_MallocAlignedBuffer
 */
OVXLIB_API void vsi_nn_FreeAlignedBuffer
    (
    uint8_t* handle
    );

OVXLIB_API vsi_bool vsi_nn_IsBufferAligned
    (
    uint8_t * buf,
    uint32_t align_start_size
    );

OVXLIB_API void vsi_nn_FormatToString
    (
    vsi_nn_tensor_t *tensor,
    char *buf,
    uint32_t buf_sz
    );

OVXLIB_API const char* vsi_nn_DescribeStatus
    (
    vsi_status status
    );

uint32_t vsi_nn_compute_filter_shape
    (
    vsi_nn_pad_e padding_type,
    uint32_t image_size,
    uint32_t ksize,
    uint32_t stride,
    uint32_t dilation_rate
    );

void vsi_nn_compute_padding
    (
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
    );

void vsi_nn_compute_padding_conv1d
    (
    uint32_t   * in_shape,
    uint32_t   * ksize,
    uint32_t   * stride,
    uint32_t   * dilation,
    vsi_nn_pad_e pad_type,
    uint32_t   * out_pad
    );

void vsi_nn_OptimizedEltOPShape
    (
       vsi_nn_tensor_t * input,
       uint32_t          sizes[VSI_NN_MAX_DIM_NUM],
       uint32_t        * num_of_dims
    );

vsi_bool vsi_nn_OptimizedEltWiseOPShape
    (
    vsi_nn_tensor_t * input0,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * output,
    uint32_t          sizes0[VSI_NN_MAX_DIM_NUM],
    uint32_t          sizes1[VSI_NN_MAX_DIM_NUM],
    uint32_t          sizes2[VSI_NN_MAX_DIM_NUM],
    uint32_t        * dim_num
    );

vsi_bool vsi_nn_IsEVISFeatureAvaiable
    (
    vsi_nn_context_t context
    );

int32_t vsi_nn_compareVersion
    (
    vsi_nn_graph_t * graph,
    uint32_t version_major,
    uint32_t version_minor,
    uint32_t version_patch
    );

typedef uint32_t(*comp_func)(void* data, int32_t left, int32_t right);

/**
 * the meta function for sort/partial sort
 * This function is the key meta function of qsort, which can be used in sort/partial sort.
 * But you can NOT use this function directly to sort/partial sort.
 * This function do NOT sort data itself, but sort its index.
 *
 * @param[in] buffer of data which will be sorted.
 * @param[in] the left(start) index of data.
 * @param[in] the right(end) index of data.
 * @param[in] compare function. the meaning of return value is as same as std::sort.
 * @param[in] recursively execute vsi_nn_partition.
 * @param[out] the sorted index of data.
 */
OVXLIB_API int32_t vsi_nn_partition
    (
        void* data,
        int32_t left,
        int32_t right,
        comp_func func,
        vsi_bool is_recursion,
        uint32_t* indices
    );

/**
 * Reorder tensors
 *
 * @param[in]  tensors Tensor list to reorder.
 * @param[in]  order New orders.
 * @param[in]  num Number of tensors.
 * @param[out] out_tensors Ordered tensors
 * */
static inline void vsi_nn_reorder_tensor
    (
    vsi_nn_tensor_t** tensors,
    const int32_t* order,
    size_t num,
    vsi_nn_tensor_t** out_tensors
    )
{
    size_t i;
    for( i = 0; i < num; i++ )
    {
        out_tensors[i] = tensors[order[i]];
    }
}

void vsi_nn_print_int_array( int32_t* array, size_t size );

float vsi_nn_activation
    (
    float value,
    vsi_nn_activation_e activation
    );

vsi_bool vsi_nn_is_same_type
    (
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    );
#ifdef __cplusplus
}
#endif

#endif
