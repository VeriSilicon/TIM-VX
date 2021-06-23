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
#include <stdlib.h>
#include <VX/vx_khr_cnn.h>
#include <VX/vx_helper.h>
#include <VX/vx.h>
#include <VX/vx_ext_program.h>

#include "vsi_nn_pub.h"
#include "utils/vsi_nn_util.h"
#include "libnnext/vsi_nn_vxkernel.h"
#include "libnnext/vx_lib_nnext.h"

#define _VX_KERNEL_VAR_CPU      (vx_client_kernel_cpu_SYNC_HOST)
#define _VX_KERNEL_ID           KERNEL_ENUM_SYNC_HOST
#define _VX_KERNEL_NAME         ("com.vivantecorp.extension.Sync_hostVXC")
#define _VX_KERNEL_FUNC_KERNEL  (vxSync_hostKernel)

static vsi_status VX_CALLBACK vxSync_hostKernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    vsi_status  status = 0;
    vx_context  context = NULL;
    vx_tensor   input = NULL;
    vx_tensor   output = NULL;
    uint8_t   * in_buffer = NULL;
    uint32_t    in_stride[8] = { 0 };
    vx_tensor_addressing in_addr = NULL;
    vsi_nn_tensor_attr_t in_attr;

    status = VX_SUCCESS;
    context = vxGetContext( (vx_reference)node );
    input  = (vx_tensor)paramObj[0];
    output = (vx_tensor)paramObj[1];
    memset(&in_attr, 0x0, sizeof(vsi_nn_tensor_attr_t));

    in_buffer = vsi_nn_ConvertRawTensorToData2( context, input,
                &in_attr, in_stride, &in_addr, VX_READ_ONLY );

    status = vsi_nn_vxCopyDataToTensor(context, output, &in_attr, in_buffer);
    if (status != VX_SUCCESS)
    {
        VSILOGE("vsi_nn_vxCopyDataToTensor failure! at line %d\n", __LINE__);
        goto OnError;
    }

OnError:
    if( NULL != in_buffer )
    {
        free( in_buffer );
    }
    return status;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
    {
    { VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    { VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED },
    };

vx_status VX_CALLBACK vxSync_hostInitializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t _VX_KERNEL_VAR_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_SYNC_HOST_list[] =
{
    &_VX_KERNEL_VAR_CPU,
    NULL
};
#ifdef __cplusplus
}
#endif
