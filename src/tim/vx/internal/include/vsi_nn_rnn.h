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
#ifndef _VSI_NN_RNN_H_
#define _VSI_NN_RNN_H_

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_internal_node.h"

#if defined(__cplusplus)
extern "C"{
#endif

/**********************************************************
* MACROS
**********************************************************/
#define VSI_NN_MAX_RNN_CONNECTION_INPUTS 16

/**********************************************************
* TYPES
**********************************************************/
typedef struct
{
    vsi_nn_tensor_id_t output;
    vsi_nn_tensor_id_t inputs[VSI_NN_MAX_RNN_CONNECTION_INPUTS];
} VSI_PUBLIC_TYPE vsi_nn_rnn_external_connection_t;

/*-------------------------------------------
Procedure to prepare input data, return FALSE
to end loop
-------------------------------------------*/
typedef vsi_bool (*vsi_nn_rnn_prepare_input_func_t)
    (
    vsi_nn_graph_t* graph,
    uint32_t iteration,
    void* user_data
    );

/*-------------------------------------------
Procedure to process output data, return FALSE
to end loop
-------------------------------------------*/
typedef vsi_bool (*vsi_rnn_rnn_process_output_func_t)
    (
    vsi_nn_graph_t* graph,
    uint32_t iteration,
    void* user_data
    );

/**********************************************************
* PUBLIC FUNCTIONS
**********************************************************/
vsi_status vsi_nn_rnn_feed_internal_state
    (
    const vsi_nn_graph_t* graph
    );

vsi_status vsi_nn_rnn_save_internal_state
    (
    const vsi_nn_graph_t* graph
    );

vsi_status vsi_nn_rnn_DeinitWksp
    (
    vsi_nn_graph_t* graph
    );

vsi_status vsi_nn_rnn_InitWksp
    (
    vsi_nn_graph_t* graph,
    const vsi_nn_rnn_external_connection_t* connections,
    uint32_t connections_count,
    void* user_data
    );

OVXLIB_API vsi_status vsi_nn_rnn_ResetBuffers
    (
    vsi_nn_graph_t* graph
    );

OVXLIB_API vsi_status vsi_nn_rnn_RunGraph
    (
    vsi_nn_graph_t* graph
    );

#if defined(__cplusplus)
}
#endif

#endif
