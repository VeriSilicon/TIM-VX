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

#include "vsi_nn_pub.h"
#include "client/vsi_nn_vxkernel.h"

#define _ARG_NUM            (0)
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _VX_KERNEL_VAR      (vx_client_kernel_SYNC_HOST)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)
#define _VSI_PARAM          (vsi_nn_client_sync_host_param)

extern vx_kernel_description_t * vx_kernel_SYNC_HOST_list[];

static void _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)inputs[i]->t;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        params[cnt] = (vx_reference)outputs[i]->t;
    }
} /* _set_inputs_outputs() */

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];

    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    /* Set inputs and outputs */
    _set_inputs_outputs( params, inputs, outputs );

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, _PARAM_NUM );

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;
    char *path = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    kernel_info.type = VX_KERNEL_TYPE_CPU;
    kernel_info.kernel = vx_kernel_SYNC_HOST_list;
    kernel_info.resource_num = 1;
    kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
    kernel_info.resource_name[0] = "sync_host";
    path = getenv("USER_VX_SOURCE_PATH");
    if(path)
        vsi_nn_VxResourceSetPath(path);

    if( kernel_info.type == VX_KERNEL_TYPE_VX)
    {
        kernel_info.kernel_index = 1;
        kernel_info.init_index = 1;
    }
    else /*kernel_info.type = VX_KERNEL_TYPE_CPU;*/
    {
        kernel_info.kernel_index = 0;
        kernel_info.init_index = 0;
    }

    self->n = vsi_nn_RegisterClientKernelAndNewNode(
            self->graph, &kernel_info);
    if (kernel_info.resource_name)
    {
        free(kernel_info.resource_name);
    }
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }
    if (NULL != op_compute_list[kernel_info.init_index])
    {
        status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
    }
    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ SYNC_HOST,
    /* init       */ NULL,
    /* compute    */ op_compute,
    /* deinit     */ vsi_nn_op_common_deinit,
    /* check      */ op_check,
    /* setup      */ vsi_nn_op_common_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cplusplus
}
#endif
