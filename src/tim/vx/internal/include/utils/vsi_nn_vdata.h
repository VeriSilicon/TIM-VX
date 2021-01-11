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
#ifndef _VSI_NN_VDATA_H
#define _VSI_NN_VDATA_H

#include <stdio.h>
#include <stdint.h>

#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

OVXLIB_API uint8_t * vsi_nn_VdataCreate
    (
    vsi_nn_graph_t * graph,
    vsi_nn_node_t  * node,
    uint32_t      * p_stream_size
    );

OVXLIB_API vsi_nn_tensor_t * vsi_nn_CreateVDataTensor
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * stream,
    vsi_nn_tensor_attr_t * attr
    );

#ifdef __cplusplus
}
#endif

#endif
