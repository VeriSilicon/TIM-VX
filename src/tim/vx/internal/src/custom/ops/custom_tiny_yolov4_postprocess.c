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
#include "vsi_nn_log.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_error.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_constraint_check.h"

typedef struct _custom_tiny_yolov4_postprocess_local_data_t {
    vx_int32 begin_dims[6][VSI_NN_MAX_DIM_NUM];
    vx_int32 end_dims[6][VSI_NN_MAX_DIM_NUM];
    vx_int32 stride_dims[VSI_NN_MAX_DIM_NUM];
} custom_tiny_yolov4_postprocess_local_data_t;

/*
 Declare number of input and output.
 */
#define _INPUT_NUM          (4)
#define _OUTPUT_NUM         (2)

static vsi_nn_internal_tensor_t *_create_internal_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tensor = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memcpy( &attr.dtype, &input->attr.dtype, sizeof( attr.dtype ) );
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    return tensor;
} /* _create_internal_tensor() */

static vsi_nn_internal_tensor_t *_create_sigmoid_internal_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tensor = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memcpy( &attr.dtype, &input->attr.dtype, sizeof( attr.dtype ) );
    if (attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC ||
        attr.dtype.qnt_type == VSI_NN_QNT_TYPE_AFFINE_SYMMETRIC)
    {
        attr.dtype.scale = 0.00390625;
        attr.dtype.zero_point = 0;
    }
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    return tensor;
} /* _create_sigmoid_internal_tensor() */

static vsi_nn_internal_tensor_t *_create_output_internal_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * output
    )
{
    vsi_nn_tensor_attr_t attr;
    vsi_nn_internal_tensor_t * tensor = NULL;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    memcpy( &attr.dtype, &output->attr.dtype, sizeof( attr.dtype ) );
    attr.dim_num = VSI_NN_DIM_AUTO;
    attr.vtl = TRUE;
    attr.is_const = FALSE;
    tensor = vsi_nn_internal_new_tensor( self, &attr, 0.0f );

    return tensor;
} /* _create_output_internal_tensor() */

static vsi_nn_internal_tensor_t *_create_strided_slice_op
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    int32_t begin_mask,
    int32_t end_mask,
    int32_t index
    )
{
    vsi_nn_custom_tiny_yolov4_postprocess_param * p = NULL;
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    p = (vsi_nn_custom_tiny_yolov4_postprocess_param *)&(self->nn_param.custom_tiny_yolov4_postprocess);

    tensor = _create_internal_tensor(self, input);
    CHECK_PTR_FAIL_GOTO( tensor, "Create internal tensor fail.", final );
    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_STRIDED_SLICE, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->node->nn_param.strided_slice.begin_dims = p->local->begin_dims[index];
    curr->node->nn_param.strided_slice.begin_dims_num = input->attr.dim_num;
    curr->node->nn_param.strided_slice.end_dims = p->local->end_dims[index];
    curr->node->nn_param.strided_slice.end_dims_num = input->attr.dim_num;
    curr->node->nn_param.strided_slice.stride_dims = p->local->stride_dims;
    curr->node->nn_param.strided_slice.stride_dims_num = input->attr.dim_num;
    curr->node->nn_param.strided_slice.begin_mask = begin_mask;
    curr->node->nn_param.strided_slice.end_mask = end_mask;
    curr->node->nn_param.strided_slice.shrink_axis_mask = 0;
    curr->node->nn_param.strided_slice.new_axis_mask = 0;
    curr->inputs[0] = input;
    curr->outputs[0] = tensor->t;
    vsi_nn_internal_setup_node( self, curr );

final:
    return tensor;
} /* _create_strided_slice() */

static vsi_nn_internal_tensor_t *_create_sigmoid_op
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input
    )
{
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    tensor = _create_sigmoid_internal_tensor(self, input);
    CHECK_PTR_FAIL_GOTO( tensor, "Create internal tensor fail.", final );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_SIGMOID, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->inputs[0] = input;
    curr->outputs[0] = tensor->t;
    vsi_nn_internal_setup_node( self, curr );

final:
    return tensor;
} /* _create_sigmoid_op() */

static vsi_nn_internal_tensor_t *_create_confidence_op
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output
    )
{
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    tensor = _create_output_internal_tensor(self, output);
    CHECK_PTR_FAIL_GOTO( tensor, "Create internal tensor fail.", final );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS_CONFIDENCE, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->inputs[0] = input;
    curr->outputs[0] = tensor->t;
    vsi_nn_internal_setup_node( self, curr );

final:
    return tensor;
} /* _create_confidence_op() */

static vsi_nn_internal_tensor_t *_create_box_op
    (
    vsi_nn_node_t *   self,
    vsi_nn_tensor_t * input0,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * output,
    float             bias0,
    float             bias1
    )
{
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;

    tensor = _create_output_internal_tensor(self, output);
    CHECK_PTR_FAIL_GOTO( tensor, "Create internal tensor fail.", final );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CUSTOM_TINY_YOLOV4_POSTPROCESS_BOX, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->inputs[0] = input0;
    curr->inputs[1] = input1;
    curr->outputs[0] = tensor->t;
    curr->node->nn_param.custom_tiny_yolov4_postprocess_box.bias_0 = bias0;
    curr->node->nn_param.custom_tiny_yolov4_postprocess_box.bias_1 = bias1;
    vsi_nn_internal_setup_node( self, curr );

final:
    return tensor;
} /* _create_box_op() */

static vsi_nn_internal_tensor_t *_create_reshape_op
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_size_t        width
    )
{
    vsi_nn_internal_tensor_t * tensor = NULL;
    vsi_nn_internal_node_t* curr = NULL;
    vsi_size_t shape_1[] = { 1, (vsi_size_t)-1, 1 };

    shape_1[0] = width;

    tensor = _create_output_internal_tensor(self, output);
    CHECK_PTR_FAIL_GOTO( tensor, "Create internal tensor fail.", final );

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_RESHAPE2, 0, 0 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->inputs[0] = input;
    curr->outputs[0] = tensor->t;
    curr->node->nn_param.reshape2.size = shape_1;
    curr->node->nn_param.reshape2.dim_num = 3;
    vsi_nn_internal_setup_node( self, curr );

final:
    return tensor;
} /* _create_reshape_op() */

static vsi_bool _create_concat_op
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input0,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * input2,
    vsi_nn_tensor_t * input3,
    vsi_nn_tensor_t * input4,
    vsi_nn_tensor_t * input5,
    vsi_nn_tensor_t * output
    )
{
    vsi_nn_internal_node_t* curr = NULL;
    vsi_bool ret = FALSE;

    curr = vsi_nn_internal_new_node( self, VSI_NN_OP_CONCAT, 6, 1 );
    CHECK_PTR_FAIL_GOTO(curr, "Create internal node failed", final);
    curr->inputs[0] = input0;
    curr->inputs[1] = input1;
    curr->inputs[2] = input2;
    curr->inputs[3] = input3;
    curr->inputs[4] = input4;
    curr->inputs[5] = input5;
    curr->outputs[0] = output;
    curr->node->nn_param.concat.axis = 1;
    ret = vsi_nn_internal_setup_node( self, curr );

final:
    return ret;
} /* _create_concat_op() */

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_compute_node( self );
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    BEGIN_IO_TYPE_DECL(CUSTOM_TINY_YOLOV4_POSTPROCESS, 4, 2)
        IO_TYPE(D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_U8|Q_ASYM,  D_U8|Q_ASYM)
    END_IO_TYPE_DECL(CUSTOM_TINY_YOLOV4_POSTPROCESS)
    if (!VALIDATE_OP_IO_TYPES(CUSTOM_TINY_YOLOV4_POSTPROCESS, self, inputs,
        self->input.num, outputs, self->output.num))
    {
        char* desc = generate_op_io_types_desc(inputs,
                self->input.num, outputs, self->output.num);
        VSILOGE("Inputs/Outputs data type not support: %s", desc);
        destroy_op_io_types_desc(desc);
        return FALSE;
    }

    return TRUE;
} /* op_check() */

static vsi_status op_optimize
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_opt_direction_e direction
    )
{
    VSI_UNREFERENCED(inputs);
    VSI_UNREFERENCED(outputs);
    return vsi_nn_internal_optimize_node( self, direction );
}

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_bool ret = FALSE;
    vsi_nn_internal_tensor_t * tensor0[12] = {NULL};
    vsi_nn_internal_tensor_t * tensor1[12] = {NULL};
    int32_t index_0 = 1;
    int32_t index_1 = 0;
    int32_t index_2 = 3;
    int32_t index_3 = 2;

    vsi_nn_internal_init_node_wksp( self );

    /**confidence**/
    /**input 0 chunk 0**/
    /*
    sub0:26x26x255 --> 26x26x81, begin: [0, 0, 4, 0] end: [0, 0, 85, 0] stride: [1, 1, 1,  1]
    sub1[26, 26, 80] = sigmoid(sub0)[26, 26, 0:0] * sigmoid(sub0)[26, 26, 1:81]
    sub2[80, 26, 26] = transpose(sub1)
    sub3[80, 676] = reshape(sub2)
    */
    tensor0[0] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 0);
    CHECK_PTR_FAIL_GOTO( tensor0[0], "Create internal tensor fail.", final );
    tensor0[1] = _create_sigmoid_op(self, tensor0[0]->t);
    CHECK_PTR_FAIL_GOTO( tensor0[1], "Create internal tensor fail.", final );
    tensor0[2] = _create_confidence_op(self, tensor0[1]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor0[2], "Create internal tensor fail.", final );
    tensor0[3] = _create_reshape_op(self, tensor0[2]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor0[3], "Create internal tensor fail.", final );
    /**chunk 1**/
    /*
    26x26x255 --> 26x26x81, begin: [0, 0, 89, 0] end: [0, 0, 170, 0] stride: [1, 1, 1,  1]
    */
    tensor0[4] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 1);
    CHECK_PTR_FAIL_GOTO( tensor0[4], "Create internal tensor fail.", final );
    tensor0[5] = _create_sigmoid_op(self, tensor0[4]->t);
    CHECK_PTR_FAIL_GOTO( tensor0[5], "Create internal tensor fail.", final );
    tensor0[6] = _create_confidence_op(self, tensor0[5]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor0[6], "Create internal tensor fail.", final );
    tensor0[7] = _create_reshape_op(self, tensor0[6]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor0[7], "Create internal tensor fail.", final );
    /**chunk 2**/
    /*
    26x26x255 --> 26x26x81, begin: [0, 0, 174, 0] end: [0, 0, 255, 0] stride: [1, 1, 1,  1]
    */
    tensor0[8] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 2);
    CHECK_PTR_FAIL_GOTO( tensor0[8], "Create internal tensor fail.", final );
    tensor0[9] = _create_sigmoid_op(self, tensor0[8]->t);
    CHECK_PTR_FAIL_GOTO( tensor0[9], "Create internal tensor fail.", final );
    tensor0[10] = _create_confidence_op(self, tensor0[9]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor0[10], "Create internal tensor fail.", final );
    tensor0[11] = _create_reshape_op(self, tensor0[10]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor0[11], "Create internal tensor fail.", final );

    /**input 1 chunk 0**/
    /*
    sub0:13x13x255 --> 26x26x81, begin: [0, 0, 4, 0] end: [0, 0, 85, 0] stride: [1, 1, 1,  1]
    sub1[13, 13, 80] = sigmoid(sub0)[13, 13, 0:0] * sigmoid(sub0)[13, 13, 1:81]
    sub2[80, 13, 13] = transpose(sub1)
    sub3[80, 169] = reshape(sub2)
    */
    tensor1[0] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 0);
    CHECK_PTR_FAIL_GOTO( tensor1[0], "Create internal tensor fail.", final );
    tensor1[1] = _create_sigmoid_op(self, tensor1[0]->t);
    CHECK_PTR_FAIL_GOTO( tensor1[1], "Create internal tensor fail.", final );
    tensor1[2] = _create_confidence_op(self, tensor1[1]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor1[2], "Create internal tensor fail.", final );
    tensor1[3] = _create_reshape_op(self, tensor1[2]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor1[3], "Create internal tensor fail.", final );
    /**chunk 1**/
    /*
    13x13x255 --> 13x13x81, begin: [0, 0, 89, 0] end: [0, 0, 170, 0] stride: [1, 1, 1,  1]
    */
    tensor1[4] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 1);
    CHECK_PTR_FAIL_GOTO( tensor1[4], "Create internal tensor fail.", final );
    tensor1[5] = _create_sigmoid_op(self, tensor1[4]->t);
    CHECK_PTR_FAIL_GOTO( tensor1[5], "Create internal tensor fail.", final );
    tensor1[6] = _create_confidence_op(self, tensor1[5]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor1[6], "Create internal tensor fail.", final );
    tensor1[7] = _create_reshape_op(self, tensor1[6]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor1[7], "Create internal tensor fail.", final );
    /**chunk 2**/
    /*
    13x13x255 --> 13x13x81, begin: [0, 0, 174, 0] end: [0, 0, 255, 0] stride: [1, 1, 1,  1]
    */
    tensor1[8] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 2);
    CHECK_PTR_FAIL_GOTO( tensor1[8], "Create internal tensor fail.", final );
    tensor1[9] = _create_sigmoid_op(self, tensor1[8]->t);
    CHECK_PTR_FAIL_GOTO( tensor1[9], "Create internal tensor fail.", final );
    tensor1[10] = _create_confidence_op(self, tensor1[9]->t, outputs[0]);
    CHECK_PTR_FAIL_GOTO( tensor1[10], "Create internal tensor fail.", final );
    tensor1[11] = _create_reshape_op(self, tensor1[10]->t, outputs[0], 80);
    CHECK_PTR_FAIL_GOTO( tensor1[11], "Create internal tensor fail.", final );

    ret = _create_concat_op(self, tensor0[3]->t, tensor0[7]->t, tensor0[11]->t,
       tensor1[3]->t, tensor1[7]->t, tensor1[11]->t, outputs[0]);
    if (ret == FALSE)
    {
        VSILOGE("Create concat operation fail");
        goto final;
    }

    ret = FALSE;
    /**box**/
    /*
    26x26x255 --> 26x26x4, begin: [0, 0, 0, 0] end: [0, 0, 4, 0] stride: [1, 1, 1,  1]
    */
    tensor0[0] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 3);
    CHECK_PTR_FAIL_GOTO( tensor0[0], "Create internal tensor fail.", final );
    tensor0[1] = _create_box_op(self, tensor0[0]->t, inputs[index_2], outputs[1], 23, 27);
    CHECK_PTR_FAIL_GOTO( tensor0[1], "Create internal tensor fail.", final );
    tensor0[2] = _create_reshape_op(self, tensor0[1]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor0[2], "Create internal tensor fail.", final );
    /*
    26x26x255 --> 26x26x4, begin: [0, 0, 85, 0] end: [0, 0, 89, 0] stride: [1, 1, 1,  1]
    */
    tensor0[3] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 4);
    CHECK_PTR_FAIL_GOTO( tensor0[3], "Create internal tensor fail.", final );
    tensor0[4] = _create_box_op(self, tensor0[3]->t, inputs[index_2], outputs[1], 37, 58);
    CHECK_PTR_FAIL_GOTO( tensor0[4], "Create internal tensor fail.", final );
    tensor0[5] = _create_reshape_op(self, tensor0[4]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor0[5], "Create internal tensor fail.", final );
    /*
    26x26x255 --> 26x26x4, begin: [0, 0, 85, 0] end: [0, 0, 89, 0] stride: [1, 1, 1,  1]
    */
    tensor0[6] = _create_strided_slice_op(self, inputs[index_0], 11, 11, 5);
    CHECK_PTR_FAIL_GOTO( tensor0[6], "Create internal tensor fail.", final );
    tensor0[7] = _create_box_op(self, tensor0[6]->t, inputs[index_2], outputs[1], 81, 82);
    CHECK_PTR_FAIL_GOTO( tensor0[7], "Create internal tensor fail.", final );
    tensor0[8] = _create_reshape_op(self, tensor0[7]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor0[8], "Create internal tensor fail.", final );

    /*
    13x13x255 --> 13x13x4, begin: [0, 0, 0, 0] end: [0, 0, 4, 0] stride: [1, 1, 1,  1]
    */
    tensor1[0] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 3);
    CHECK_PTR_FAIL_GOTO( tensor1[0], "Create internal tensor fail.", final );
    tensor1[1] = _create_box_op(self, tensor1[0]->t, inputs[index_3], outputs[1], 81, 82);
    CHECK_PTR_FAIL_GOTO( tensor1[1], "Create internal tensor fail.", final );
    tensor1[2] = _create_reshape_op(self, tensor1[1]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor1[2], "Create internal tensor fail.", final );
    /*
    13x13x255 --> 13x13x4, begin: [0, 0, 85, 0] end: [0, 0, 89, 0] stride: [1, 1, 1,  1]
    */
    tensor1[3] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 4);
    CHECK_PTR_FAIL_GOTO( tensor1[3], "Create internal tensor fail.", final );
    tensor1[4] = _create_box_op(self, tensor1[3]->t, inputs[index_3], outputs[1], 135, 169);
    CHECK_PTR_FAIL_GOTO( tensor1[4], "Create internal tensor fail.", final );
    tensor1[5] = _create_reshape_op(self, tensor1[4]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor1[5], "Create internal tensor fail.", final );
    /*
    13x13x255 --> 13x13x4, begin: [0, 0, 170, 0] end: [0, 0, 174, 0] stride: [1, 1, 1,  1]
    */
    tensor1[6] = _create_strided_slice_op(self, inputs[index_1], 11, 11, 5);
    CHECK_PTR_FAIL_GOTO( tensor1[6], "Create internal tensor fail.", final );
    tensor1[7] = _create_box_op(self, tensor1[6]->t, inputs[index_3], outputs[1], 344, 319);
    CHECK_PTR_FAIL_GOTO( tensor1[7], "Create internal tensor fail.", final );
    tensor1[8] = _create_reshape_op(self, tensor1[7]->t, outputs[1], 4);
    CHECK_PTR_FAIL_GOTO( tensor1[8], "Create internal tensor fail.", final );

    ret = _create_concat_op(self, tensor0[2]->t, tensor0[5]->t, tensor0[8]->t,
        tensor1[2]->t, tensor1[5]->t, tensor1[8]->t, outputs[1]);
    if (ret == FALSE)
    {
        VSILOGE("Create concat operation fail");
        goto final;
    }

final:
    return ret;
} /* op_setup() */

static vsi_status op_init
    (
    vsi_nn_node_t* self
    )
{
    int32_t i = 0;
    vsi_nn_custom_tiny_yolov4_postprocess_param *p = &self->nn_param.custom_tiny_yolov4_postprocess;
    p->local = \
        (custom_tiny_yolov4_postprocess_local_data_t*)malloc(sizeof(custom_tiny_yolov4_postprocess_local_data_t));
    CHECK_PTR_FAIL_GOTO(p->local, "create buffer fail", final);
    memset(p->local, 0, sizeof(custom_tiny_yolov4_postprocess_local_data_t));
    for ( i = 0; i < VSI_NN_MAX_DIM_NUM; i++ )
    {
        p->local->stride_dims[i] = 1;
    }
    p->local->begin_dims[0][2] = 4;
    p->local->end_dims[0][2] = 85;

    p->local->begin_dims[1][2] = 89;
    p->local->end_dims[1][2] = 170;

    p->local->begin_dims[2][2] = 174;
    p->local->end_dims[2][2] = 255;

    p->local->begin_dims[3][2] = 0;
    p->local->end_dims[3][2] = 4;

    p->local->begin_dims[4][2] = 85;
    p->local->end_dims[4][2] = 89;

    p->local->begin_dims[5][2] = 170;
    p->local->end_dims[5][2] = 174;
final:
    return VSI_SUCCESS;
} /* op_init() */

static vsi_status op_deinit
    (
    vsi_nn_node_t* self
    )
{
    vsi_status status = VSI_SUCCESS;

    status = vsi_nn_op_common_deinit(self);

    vsi_nn_safe_free(self->nn_param.custom_tiny_yolov4_postprocess.local);
    vsi_nn_internal_deinit_node_wksp( self );

    return status;
} /* op_deinit() */

__BEGIN_DECLS

/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ CUSTOM_TINY_YOLOV4_POSTPROCESS,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ op_optimize,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );

__END_DECLS

