/****************************************************************************
*
*    Copyright (c) 2022 Vivante Corporation
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
#ifndef _VSI_NN_TYPES_PRV_H_
#define _VSI_NN_TYPES_PRV_H_

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"

#if defined(__cplusplus)
extern "C"{
#endif

typedef struct _vsi_nn_swap_handle_cache_item
{
    vsi_nn_link_list_t link_list;
    vsi_nn_node_t* node;
    vsi_nn_tensor_t* tensor;
    uint32_t idx;
} vsi_nn_swap_handle_cache_item_t;

typedef struct _vsi_nn_swap_handle_cache
{
    int8_t is_feature_on;
    int8_t is_cached;
    vsi_nn_swap_handle_cache_item_t* cache_list;
} vsi_nn_swap_handle_cache_t;

/**
 * Internal Graph structure, internal use only.
 */
typedef struct _vsi_nn_graph_prv
{
    /** Public Ovxlib Graph(pot)*/
    vsi_nn_graph_t pog;

    // Add graph internal attribute here...
    vsi_nn_swap_handle_cache_t swap_handle_cache;
    vsi_nn_runtime_option_t* options;
} vsi_nn_graph_prv_t;

/** Internal Node structure, internal use only. */
typedef struct _vsi_nn_node_prv
{
    /** Public Ovxlib Node(pon)*/
    vsi_nn_node_t pon;

    /** some op will adjust its const tensors in op_setup,
     * this field is to indicate whether the const tensors
     * are processed in op_setup phase, this adjustment cannot
     * be done more than once */
    int8_t processed;

    // Add node internal attribute here...
#if VX_GRAPH_BATCH_OPT_SUPPORT
    /*split the node to "split_num" on batch dim.*/
    vsi_size_t split_num;
#endif
} vsi_nn_node_prv_t;

/**
    * Internal Tensor structure, internal use only.
    */
typedef struct _vsi_nn_tensor_prv
{
    /** Public Ovxlib Tensor(pot)*/
    vsi_nn_tensor_t pot;

    /** Tensor handle*/
    uint8_t* handle;

    /** is scalar*/
    int8_t is_scalar;

    /** some op will adjust its const tensors in op_setup,
     * this field is to indicate whether the const tensors
     * are processed in op_setup phase, this adjustment cannot
     * be done more than once */
    int8_t processed;

    /** For mapping tensor patch.
    *   map_id The address of a vx_map_id variable where the function returns a map identifier.
    */
    vx_map_id map_id;

    /** create tensor from axisram.*/
    int8_t is_from_axisram;

    /** 2:4 sparsity attr. */
#if defined(VSI_TENSOR_SPARSITY_SUPPORT)
    vx_tensor_sparsity_param_e sparsity_type; /*!< \brief sparsity type for the tensor */
#endif

    // Add tensor internal attribute here...
} vsi_nn_tensor_prv_t;

#if defined(__cplusplus)
}
#endif

#endif
