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
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "utils/vsi_nn_math.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_log.h"
#include "kernel/vsi_nn_kernel.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel_gpu_shape_optimize.h"
#include "utils/vsi_nn_constraint_check.h"
#include "utils/vsi_nn_dtype_util.h"
#include "vsi_nn_error.h"

#define _INPUT_NUM          (2)
#define _OUTPUT_NUM         (1)

#define VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS 2

static vsi_nn_tensor_t* _expand_scale_tensor
    (
    vsi_nn_graph_t  *graph,
    vsi_nn_tensor_t *scale,
    vsi_size_t          scale_size_in,
    vsi_size_t          scale_size_out
    )
{
    vsi_status status = VX_SUCCESS;
    float* f32_in_buffer   = NULL;
    float* f32_out_buffer  = NULL;
    vsi_size_t  i = 0;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_tensor_t*  scale_tensor  = NULL;
    vsi_nn_dtype_t   out_dtype;

    f32_out_buffer= (float *)malloc(scale_size_out * sizeof(float));
    CHECK_PTR_FAIL_GOTO( f32_out_buffer, "Create buffer fail.", final );
    memset(f32_out_buffer, 0, scale_size_out * sizeof(float));
    f32_in_buffer = vsi_nn_ConvertTensorToFloat32Data(graph, scale);
    if (NULL == f32_in_buffer)
    {
        scale_tensor = NULL;
        goto final;
    }

    for (i = 0; i < scale_size_in; i++)
    {
        f32_out_buffer[i] = f32_in_buffer[i];
    }
    for (i = scale_size_in; i < scale_size_out; i++)
    {
        f32_out_buffer[i] = f32_in_buffer[scale_size_in - 1];
    }

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    attr.size[0] = scale_size_out;
    attr.size[1] = 1;
    attr.dim_num = 2;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
    attr.vtl = FALSE;
    scale_tensor = vsi_nn_CreateTensor(graph, &attr);
    CHECK_PTR_FAIL_GOTO( scale_tensor, "Create tensor fail.", final );
    out_dtype          = scale->attr.dtype;
    out_dtype.vx_type  = VSI_NN_TYPE_FLOAT32;
    out_dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    status = vsi_nn_CopyRawDataToTensor (graph,
          (uint8_t*)f32_out_buffer, &out_dtype, scale_tensor);
    if (VSI_SUCCESS != status)
    {
        goto final;
    }

final:
    if (f32_in_buffer)
    {
        free(f32_in_buffer);
        f32_in_buffer = NULL;
    }

    if (f32_out_buffer)
    {
        free(f32_out_buffer);
        f32_out_buffer = NULL;
    }

    return scale_tensor;
}

static vsi_bool _check_value_is_equal_to_one
    (
    vsi_nn_graph_t* graph,
    vsi_nn_tensor_t* tensor
    )
{
    vsi_bool ret = TRUE;
    float* tensor_data = NULL;
    vsi_size_t elements = 0;
    vsi_size_t i = 0;

    elements = vsi_nn_GetElementNum( tensor );
    tensor_data = vsi_nn_ConvertTensorToFloat32Data( graph, tensor );
    if ( NULL == tensor_data )
    {
        VSILOGE( "Convert data fail." );
        return FALSE;
    }

    for (i = 0; i < elements; i++)
    {
        if ( vsi_abs(tensor_data[i] - 1.0f) > 1e-5 )
        {
            ret = FALSE;
            break;
        }
    }

    if ( !tensor->attr.is_created_from_handle )
    {
        if ( tensor_data )
        {
            free(tensor_data);
        }
    }

    return ret;
}

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status      = VSI_FAILURE;
    int32_t    axis        = 0;
    int32_t    new_axis    = 0;
    uint32_t   axis_size   = 0;
    uint32_t   rank_in     = 0;
    uint32_t   rank_out    = 0;
    vsi_size_t   size        = 1;
    uint32_t   i           = 0;
    vsi_size_t   scale_size  = 1;
    vsi_size_t shapes[3][VSI_NN_MAX_DIM_NUM] = {{ 1 }};
    vsi_nn_tensor_t* reshape_tensors[3] = { NULL };
    vsi_nn_l2normalizescale_param * p = NULL;
    vsi_bool ret = FALSE;
    vsi_nn_kernel_param_t * param  = NULL;
    vsi_bool is_expand_scale = vx_false_e;

    p = &(self->nn_param.l2normalizescale);
    axis = p->axis;

    if ( (inputs[1]->attr.is_const == TRUE && _check_value_is_equal_to_one(self->graph, inputs[1])) ||
        ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 &&
          outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 )
        )
    {
        return vsi_nn_internal_compute_node( self );
    }

    param =vsi_nn_kernel_param_create();

    ret = vsi_nn_kernel_optimize_reduce_shape(
            inputs[0]->attr.size, inputs[0]->attr.dim_num,
            &axis, 1,
            outputs[0]->attr.size, outputs[0]->attr.dim_num,
            shapes[0], &rank_in, shapes[2], &rank_out,
            &new_axis, &axis_size);
    size = inputs[1]->attr.size[0];
    for (i = 1; i < inputs[1]->attr.dim_num; i ++)
    {
        size *= inputs[1]->attr.size[i];
    }
    shapes[1][0] = size;
    shapes[1][1] = 1;
    shapes[1][2] = 1;
    shapes[1][3] = 1;
    scale_size = shapes[0][new_axis];
    is_expand_scale = (vx_bool)(TRUE == inputs[1]->attr.is_const);
    vsi_nn_kernel_param_add_int32( param, "axis",  new_axis );

    if ( ret )
    {
        reshape_tensors[0] = vsi_nn_reshape_tensor( self->graph,
                inputs[0], shapes[0], rank_in );
        if (is_expand_scale)
        {
            reshape_tensors[1] = _expand_scale_tensor(self->graph, inputs[1], size, scale_size);
        }
        else
        {
            reshape_tensors[1] = vsi_nn_reshape_tensor( self->graph,
                    inputs[1], shapes[1], 2 );
        }
        reshape_tensors[2] = vsi_nn_reshape_tensor( self->graph,
                outputs[0], shapes[0], rank_in );

        self->n = (vx_node)vsi_nn_kernel_selector( self->graph,
                "l2normalizescale",
                &reshape_tensors[0], _INPUT_NUM,
                &reshape_tensors[2], _OUTPUT_NUM, param );

        vsi_nn_ReleaseTensor( &reshape_tensors[0] );
        vsi_nn_ReleaseTensor( &reshape_tensors[1] );
        vsi_nn_ReleaseTensor( &reshape_tensors[2] );
    }
    if( self->n )
    {
        status = VSI_SUCCESS;
    }

    vsi_nn_kernel_param_release( &param );

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(L2NORMALIZESCALE, _INPUT_NUM, _OUTPUT_NUM)
        IO_TYPE(D_F16,        D_F16,  D_F16)
        IO_TYPE(D_F16,        D_F32,  D_F16)
        IO_TYPE(D_BF16,       D_BF16, D_BF16)
        IO_TYPE(D_BF16,       D_F32,  D_BF16)
        IO_TYPE(D_I8|Q_DFP,   D_F16,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F16,  D_F16)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_I8|Q_DFP)
        IO_TYPE(D_I8|Q_DFP,   D_F32,  D_F16)
        IO_TYPE(D_U8|Q_ASYM,  D_F16,  D_U8|Q_ASYM)
        IO_TYPE(D_U8|Q_ASYM,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F16,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_I16|Q_DFP)
        IO_TYPE(D_I16|Q_DFP,  D_F32,  D_F16)
        IO_TYPE(D_F32,        D_F32,  D_F32)
        IO_TYPE(D_U8|Q_ASYM,  D_F32,  D_U8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_F16,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_F16,  D_F16)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_I8|Q_ASYM)
        IO_TYPE(D_I8|Q_ASYM,  D_F32,  D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_F16,  D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_SYM,   D_F16,  D_F16)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_I8|Q_SYM)
        IO_TYPE(D_I8|Q_SYM,   D_F32,  D_F16)
        IO_TYPE(D_I16|Q_ASYM, D_F16,  D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM, D_F16,  D_F16)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_I16|Q_ASYM)
        IO_TYPE(D_I16|Q_ASYM, D_F32,  D_F16)
        IO_TYPE(D_I16|Q_SYM,  D_F16,  D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_SYM,  D_F16,  D_F16)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_I16|Q_SYM)
        IO_TYPE(D_I16|Q_SYM,  D_F32,  D_F16)
    END_IO_TYPE_DECL(L2NORMALIZESCALE)
    if (!VALIDATE_OP_IO_TYPES(L2NORMALIZESCALE, self, inputs, self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    uint32_t i;
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        if (self->nn_param.l2normalizescale.local.local_tensor[i] != NULL)
        {
            vxReleaseTensor(&(self->nn_param.l2normalizescale.local.local_tensor[i]));
            self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
        }
    }

    vsi_nn_internal_deinit_node_wksp( self );

    vsi_nn_op_common_deinit(self);

    return VSI_SUCCESS;
} /* op_deinit() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = TRUE;
    vsi_nn_internal_node_t* curr = NULL;

    if( NULL == self )
    {
        return FALSE;
    }

    vsi_nn_internal_init_node_wksp( self );

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        self->nn_param.l2normalizescale.axis += (int32_t)inputs[0]->attr.dim_num;
    }

    if (self->nn_param.l2normalizescale.axis < 0)
    {
        VSILOGD("l2normalizescale Invalid Axis: %d", self->nn_param.l2normalizescale.axis);
        return FALSE;
    }

    if ( inputs[1]->attr.is_const == TRUE && _check_value_is_equal_to_one( self->graph, inputs[1] ) )
    {
        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_L2_NORMALIZE, 0, 0);
        curr->node->nn_param.l2_normalize.axis = self->nn_param.l2normalizescale.axis;
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node( self, curr );
    }
    else if ( inputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 &&
        outputs[0]->attr.dtype.vx_type == VSI_NN_TYPE_BFLOAT16 )
    {
        vsi_nn_internal_tensor_t* output_tensor = NULL;
        vsi_nn_internal_tensor_t* reshape_tensor = NULL;
        vsi_nn_tensor_attr_t attr;
        int32_t dim_num = inputs[0]->attr.dim_num;
        int32_t i = 0;

        memcpy( &attr, &outputs[0]->attr, sizeof( attr ) );
        attr.vtl = TRUE;
        attr.is_const = FALSE;
        output_tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_L2_NORMALIZE, 0, 0);
        curr->node->nn_param.l2_normalize.axis = self->nn_param.l2normalizescale.axis;
        curr->inputs[0] = inputs[0];
        curr->outputs[0] = output_tensor->t;
        vsi_nn_internal_setup_node( self, curr );

        memcpy( &attr, &inputs[1]->attr, sizeof( attr ) );
        for (i = 0; i < dim_num; i++)
        {
            attr.size[i] = i == self->nn_param.l2normalizescale.axis ? inputs[0]->attr.size[i] : 1;
        }
        attr.dim_num = dim_num;
        if (attr.dtype.vx_type != VSI_NN_TYPE_BFLOAT16)
        {
            attr.dtype.vx_type = VSI_NN_TYPE_BFLOAT16;
            attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        }
        reshape_tensor = vsi_nn_internal_new_tensor(self, &attr, 0.0f);
        vsi_nn_ConvertTensor(self->graph, inputs[1], reshape_tensor->t);

        curr = vsi_nn_internal_new_node(self, VSI_NN_OP_MULTIPLY, 0, 0);
        curr->inputs[0] = output_tensor->t;
        curr->inputs[1] = reshape_tensor->t;
        curr->node->nn_param.multiply.scale = 1.0f;
        curr->node->vx_param.overflow_policy = VX_CONVERT_POLICY_SATURATE;
        curr->node->vx_param.rounding_policy = VX_ROUND_POLICY_TO_NEAREST_EVEN;
        curr->outputs[0] = outputs[0];
        vsi_nn_internal_setup_node( self, curr );
    }
    else
    {
        ret = vsi_nn_op_common_setup(self, inputs, outputs);
    }

    return ret;
}

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    uint32_t  i = 0;

    if (vsi_nn_compareVersion(self->graph, 1, 1, 13) == -1)
    {
        self->nn_param.l2normalizescale.axis = VSI_NN_L2NORMALIZESCALE_DEFAULT_AXIS;
    }
    for (i = 0; i < _VSI_NN_L2NORMALIZESCALE_LOCAL_TENSOR_NUM; i++)
    {
        self->nn_param.l2normalizescale.local.local_tensor[i] = NULL;
    }

    return status;
} /* op_init() */

#ifdef __cplusplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ L2NORMALIZESCALE,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ 2,
    /* output_num */ 1
    );
#ifdef __cplusplus
}
#endif
