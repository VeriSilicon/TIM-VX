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
#ifndef _VSI_NN_TENSOR_UTIL_PRV_H
#define _VSI_NN_TENSOR_UTIL_PRV_H

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "vsi_nn_types.h"
#include "vsi_nn_types_prv.h"

#ifdef __cplusplus
extern "C" {
#endif

#define vsi_safe_release_node(_n) if(_n){vxReleaseNode( ((vx_node*)&_n) ); _n = NULL;}

#define VSI_NN_SUPPORT_LSTM_GRU_SP_IMPL      (1)
/**
 * Get tensor handle
 *
 * @param[in] tensor, a pointer pointed to vsi_nn_tensor_prv_t.
 *
 * @return handle of tensor on success, or NULL otherwise.
 */
uint8_t* _get_tensor_handle
    (
    vsi_nn_tensor_prv_t* tensor
    );

/**
 * Set tensor handle
 *
 * @param[in] tensor, a pointer pointed to vsi_nn_tensor_prv_t.
 * @param[in] handle, a handle need to be set to tensor.
 *
 * @return VSI_SUCCESS on success, or VSI_FAILURE otherwise.
 */
vsi_status _set_tensor_handle
    (
    vsi_nn_tensor_prv_t* tensor,
    uint8_t*             handle
    );

int8_t _get_tensor_is_scalar
    (
    vsi_nn_tensor_prv_t* tensor
    );

vsi_status _set_tensor_is_scalar
    (
    vsi_nn_tensor_prv_t* tensor,
    int8_t is_salar
    );

/**
 * Create a new dummy tensor
 * Create a new dummy tensor with given attributes.
 *
 * @param[in] graph Graph handle
 * @param[in] attr Tensor attributes
 *
 * @return Tensor handle on success, or NULL otherwise.
 */
vsi_nn_tensor_t * vsi_nn_create_dummy_tensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    );

vsi_bool vsi_nn_is_stream_process_supported_types
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t** inputs,
    size_t input_num
    );

vsi_bool vsi_nn_is_same_data_type(
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    );

vsi_bool vsi_nn_is_same_quant_type(
    vsi_nn_tensor_t * src,
    vsi_nn_tensor_t * dst
    );

#ifdef __cplusplus
}
#endif

#endif
