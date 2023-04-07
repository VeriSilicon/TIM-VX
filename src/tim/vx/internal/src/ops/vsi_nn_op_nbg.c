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

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_log.h"

static const char *_get_vx_nbg_type
    (
    vsi_nn_nbg_type_e type
    )
{
    switch (type)
    {
    case VSI_NN_NBG_FILE:
        return VX_VIVANTE_IMPORT_KERNEL_FROM_FILE;
    case VSI_NN_NBG_FOLDER:
        return VX_VIVANTE_IMPORT_KERNEL_FROM_FOLDER;
    case VSI_NN_NBG_LABEL:
        return VX_VIVANTE_IMPORT_KERNEL_FROM_LABEL;
    case VSI_NN_NBG_POINTER:
        return VX_VIVANTE_IMPORT_KERNEL_FROM_POINTER;
    default:
        VSILOGE("error nbg type %d", type);
        return NULL;
    }
}

static void _set_io_index
(
    vsi_nn_node_t* self,
    vsi_nn_tensor_t** inputs,
    vsi_nn_tensor_t** outputs
)
{
    uint32_t idx, i, j;

    idx = 0;
    for (i = 0; i < self->input.num; i++)
    {
        uint32_t scalar_index=0;
        vx_parameter param = 0;
        vx_enum type = 0;

        vxSetParameterByIndex(self->n, idx++, (vx_reference)inputs[i]->t);
        scalar_index = idx;
        param = vxGetParameterByIndex(self->n, scalar_index);
        vxQueryParameter(param, VX_PARAMETER_TYPE, &type, sizeof(vx_enum));
        if (param != NULL)
        {
            vxReleaseParameter(&param);
            param = NULL;

        }
        if (type != VX_TYPE_SCALAR)
        {
            continue;
        }
        else
        {

            /* 4 crop scalar parameters input */
            for (j = scalar_index; j < scalar_index + 4; j++)
            {
                vx_enum data_type = 0;
                vx_reference ref = 0;
                vsi_status status;
                param = vxGetParameterByIndex(self->n, j);
                vxQueryParameter(param, VX_PARAMETER_REF, &ref, sizeof(vx_reference));
                status = vxQueryScalar((vx_scalar)ref, VX_SCALAR_TYPE, &data_type, sizeof(vx_enum));
                if (status == VX_ERROR_INVALID_REFERENCE)
                {
                    vx_scalar scalar = vxCreateScalar(self->graph->ctx->c, VX_TYPE_INT32, 0);
                    ref = (vx_reference)scalar;
                    vxSetParameterByIndex(self->n, idx++, ref);
                    vxReleaseReference(&ref);
                }
                if (param != NULL)
                {
                    vxReleaseParameter(&param);
                    param = NULL;
                }
            }
        }
    }
    for (i = 0; i < self->output.num; i++)
    {
        vxSetParameterByIndex(self->n, idx++, (vx_reference)outputs[i]->t);
    }
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vx_node node;
    vx_kernel kernel;

    status = VSI_FAILURE;
    kernel = NULL;
    kernel = vxImportKernelFromURL(
        self->graph->ctx->c,
        _get_vx_nbg_type(self->nn_param.nbg.type),
        self->nn_param.nbg.url
        );
    if(NULL == kernel)
    {
        return status;
    }
    self->nn_param.nbg.local.kernel = kernel;

    node = NULL;
    node = vxCreateGenericNode(
        self->graph->g,
        self->nn_param.nbg.local.kernel
        );
    if(NULL == node)
    {
        vxRemoveKernel(kernel);
        return status;
    }

    self->nn_param.nbg.local.kernel = kernel;
    self->n = node;
    _set_io_index(self, inputs, outputs);
    status = VSI_SUCCESS;

    return status;
}

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*
     * Network Binary Graph node do not need to calculate output shape
     */
    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vx_kernel kernel;

    kernel = self->nn_param.nbg.local.kernel;
    if(kernel)
    {
        vxRemoveKernel(kernel);
        kernel = self->nn_param.nbg.local.kernel = NULL;
    }
    vsi_nn_op_common_deinit(self);
    return VSI_SUCCESS;
} /* op_deinit() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ NBG,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 10,
    /* output_num */ 10
    );
#ifdef __cplusplus
}
#endif
