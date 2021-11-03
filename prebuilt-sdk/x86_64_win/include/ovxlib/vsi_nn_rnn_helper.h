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
#ifndef _VSI_NN_RNN_HELPER_H
#define _VSI_NN_RNN_HELPER_H

#include "vsi_nn_types.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_internal_node.h"
#include "utils/vsi_nn_util.h"

#if defined(__cplusplus)
extern "C"{
#endif

/**
 * find the best kernel size on HW
 *
 * @param[in] is multi batch.
 * @param[in] input size.
 * @param[out] the height of the best kernel size.
 * @param[out] the width of the best kernel size.
 */
vsi_bool vsi_nn_rnn_find_best_kernel_size
    (
    vsi_bool multi_batch,
    uint32_t input_size,
    uint32_t* p_kernel_h,
    uint32_t* p_kernel_w
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_input_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_process_output_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    );

vsi_bool vsi_nn_rnn_process_output_for_nn_fc2
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_bool multi_batch,
    uint32_t kernel_h,
    uint32_t kernel_w,
    int32_t use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tp_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    uint32_t kernel_h,
    uint32_t kernel_w,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_prepare_weight_for_nn_fc
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * weight,
    uint32_t kernel_h,
    uint32_t kernel_w
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_nn_fc_relu
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * weight,
    vsi_nn_tensor_t * bias,
    uint32_t kernel_h,
    uint32_t kernel_w,
    vsi_bool has_relu,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tensor_add
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input1,
    vsi_nn_tensor_t * input2,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_op_t vsi_nn_rnn_get_act_op_type
    (
    vsi_nn_activation_e type
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_activation
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_activation_e act_type,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_transpose_time_major
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    vsi_bool use_virtual_tensor
    );

void vsi_nn_rnn_split_input_tensor
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t ** output,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    );

void vsi_nn_rnn_data_check_aligned
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** input,
    uint32_t time_step,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_split_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_reshape_cell_output
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t * input,
    uint32_t batch_size,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_binary_operator
    (
    vsi_nn_node_t* self,
    vsi_nn_op_t op,
    vsi_nn_tensor_t* operand1,
    vsi_nn_tensor_t* operand2,
    const vsi_nn_dtype_t* output_dtype,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_concat_impl
    (
    vsi_nn_node_t* self,
    uint32_t axis,
    vsi_bool use_virtual_tensor,
    vsi_nn_tensor_t* tensor,
    ...
    );
#define vsi_nn_rnn_create_concat(_node, _axis, _virtual, _tensor, ...) \
    vsi_nn_rnn_create_concat_impl(_node, _axis, _virtual, _tensor, __VA_ARGS__, END_OF_VARIADIC_ARGUMENTS)

vsi_nn_internal_tensor_t** vsi_nn_create_split
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* tensor,
    uint32_t axis,
    uint32_t slices_num,
    uint32_t* slices,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_reshape
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_size_t* size,
    vsi_size_t dim_num,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_permute
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_size_t* perm,
    vsi_size_t dim_num,
    vsi_bool use_virtual_tensor
    );

vsi_nn_internal_tensor_t* vsi_nn_rnn_create_tensor_copy
    (
    vsi_nn_node_t* self,
    vsi_nn_tensor_t* input_tensor,
    vsi_nn_tensor_t* output_tensor,
    vsi_nn_dtype_t* dtype,
    vsi_bool use_virtual_tensor
    );

#if defined(__cplusplus)
}
#endif

#endif /* _VSI_NN_RNN_HELPER_H */