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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_error.h"

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t * type_tensor = NULL;
    vx_nn_lshproj_params_t p;
    vx_bool valued = TRUE;
    vsi_nn_tensor_t * weight_tensor = NULL;
    float* const_data = NULL;

    type_tensor = vsi_nn_VariableToTensor(self,
        (uint8_t *)&self->nn_param.lsh_projection.type,
        VSI_NN_TYPE_INT32);
    CHECK_PTR_FAIL_GOTO( type_tensor, "Create tensor fail.", final );

    memset(&p, 0, sizeof(p));
    p.hash_func = REQUIRED_IO(inputs[0]);
    p.weights = OPTIONAL_IO(inputs[2]);
    //p.weights = inputs[2]->t;
    p.type = type_tensor->t;
    //This is a hack
    // Need driver fix this
    if (p.weights == NULL)
    {
        vsi_nn_tensor_attr_t attr;
        float const_one = 1.0;
        vsi_size_t i;
        vsi_size_t count = inputs[1]->attr.size[1];

        const_data = (float*)malloc(count * sizeof(float));
        CHECK_PTR_FAIL_GOTO( const_data, "Create buffer fail.", final );

        for (i = 0; i < count; i++)
        {
            const_data[i] = const_one;
        }
        memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
        attr.size[0] = count;
        attr.dim_num = 1;
        attr.is_const = TRUE;
        attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
        weight_tensor = vsi_nn_CreateTensorFromData(self->graph,
            (uint8_t *)const_data, &attr);
        CHECK_PTR_FAIL_GOTO( weight_tensor, "Create tensor fail.", final );
        p.weights = weight_tensor->t;
    }
    vxSetTensorAttribute(p.weights, VX_TENSOR_VALUE, &valued, sizeof(vx_bool));

    self->n = vxLSHProjectionLayer( self->graph->g,
            inputs[1]->t, &p, sizeof(p), outputs[0]->t);
    if( !self->n )
    {
        status = VSI_FAILURE;
    }

final:
    vsi_nn_safe_free(const_data);
    vsi_safe_release_tensor( type_tensor );
    vsi_safe_release_tensor( weight_tensor );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(self);
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * node,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    if( VSI_NN_DIM_AUTO == outputs[0]->attr.dim_num )
    {
        outputs[0]->attr.dim_num = 1;
        if( VSI_NN_LSH_PROJECTION_SPARSE == node->nn_param.lsh_projection.type )
        {
            outputs[0]->attr.size[0] = inputs[0]->attr.size[1];
        }
        else if( VSI_NN_LSH_PROJECTION_DENSE == node->nn_param.lsh_projection.type )
        {
            outputs[0]->attr.size[0] = (uint32_t)vsi_nn_GetElementNum( inputs[0] );
        }
        else
        {
            VSILOGE("Unknown lsh projection hash type.");
        }
    }
    return TRUE;
} /* op_setup() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSH_PROJECTION,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 3,
    /* output_num */ 1
    );
#ifdef __cpluplus
}
#endif
