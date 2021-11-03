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
/** @file */
#ifndef _VSI_NN_NODE_TYPES_H_
#define _VSI_NN_NODE_TYPES_H_

#include "vsi_nn_types.h"
#include "vsi_nn_assert.h"
#include "ops/vsi_nn_op_activations.h"
#include "ops/vsi_nn_op_batch_norm.h"
#include "ops/vsi_nn_op_multiply.h"
#include "ops/vsi_nn_op_concat.h"
#include "ops/vsi_nn_op_split.h"
#include "ops/vsi_nn_op_conv2d.h"
#include "ops/vsi_nn_op_deconvolution.h"
#include "ops/vsi_nn_op_fullconnect.h"
#include "ops/vsi_nn_op_lrn.h"
#include "ops/vsi_nn_op_permute.h"
#include "ops/vsi_nn_op_pool.h"
#include "ops/vsi_nn_op_proposal.h"
#include "ops/vsi_nn_op_reshape.h"
#include "ops/vsi_nn_op_roi_pool.h"
#include "ops/vsi_nn_op_upsample.h"
#include "ops/vsi_nn_op_resize.h"
#include "ops/vsi_nn_op_lstm.h"
#include "ops/vsi_nn_op_reorg.h"
#include "ops/vsi_nn_op_l2normalizescale.h"
#include "ops/vsi_nn_op_crop.h"
#include "ops/vsi_nn_op_relun.h"
#include "ops/vsi_nn_op_divide.h"
#include "ops/vsi_nn_op_tanh.h"
#include "ops/vsi_nn_op_dropout.h"
#include "ops/vsi_nn_op_shufflechannel.h"
#include "ops/vsi_nn_op_prelu.h"
#include "ops/vsi_nn_op_elu.h"
#include "ops/vsi_nn_op_reverse.h"
#include "ops/vsi_nn_op_space2depth.h"
#include "ops/vsi_nn_op_space2depth_internal.h"
#include "ops/vsi_nn_op_depth2space.h"
#include "ops/vsi_nn_op_depth2space_internal.h"
#include "ops/vsi_nn_op_maximum.h"
#include "ops/vsi_nn_op_scale.h"
#include "ops/vsi_nn_op_slice.h"
#include "ops/vsi_nn_op_space2batch.h"
#include "ops/vsi_nn_op_batch2space.h"
#include "ops/vsi_nn_op_pad.h"
#include "ops/vsi_nn_op_imageprocess.h"
#include "ops/vsi_nn_op_matrixmul.h"
#include "ops/vsi_nn_op_lstmunit.h"
#include "ops/vsi_nn_op_layernormalize.h"
#include "ops/vsi_nn_op_reduce.h"
#include "ops/vsi_nn_op_softmax.h"
#include "ops/vsi_nn_op_instancenormalize.h"
#include "ops/vsi_nn_op_tensorstackconcat.h"
#include "ops/vsi_nn_op_strided_slice.h"
#include "ops/vsi_nn_op_signalframe.h"
#include "ops/vsi_nn_op_argmax.h"
#include "ops/vsi_nn_op_svdf.h"
#include "ops/vsi_nn_op_conv1d.h"
#include "ops/vsi_nn_op_nbg.h"
#include "ops/vsi_nn_op_spatial_transformer.h"
#include "ops/vsi_nn_op_logical_ops.h"
#include "ops/vsi_nn_op_select.h"
#include "ops/vsi_nn_op_concatshift.h"
#include "ops/vsi_nn_op_relational_ops.h"
#include "ops/vsi_nn_op_pow.h"
#include "ops/vsi_nn_op_floordiv.h"
#include "ops/vsi_nn_op_lstmunit_activation.h"
#include "ops/vsi_nn_op_lstmunit_ovxlib.h"
#include "ops/vsi_nn_op_tensor_add_mean_stddev_norm.h"
#include "ops/vsi_nn_op_lstm_ovxlib.h"
#include "ops/vsi_nn_op_lsh_projection.h"
#include "ops/vsi_nn_op_rnn.h"
#include "ops/vsi_nn_op_stack.h"
#include "ops/vsi_nn_op_floor.h"
#include "ops/vsi_nn_op_neg.h"
#include "ops/vsi_nn_op_exp.h"
#include "ops/vsi_nn_op_clip.h"
#include "ops/vsi_nn_op_pre_process_tensor.h"
#include "ops/vsi_nn_op_post_process.h"
#include "ops/vsi_nn_op_pre_process_gray.h"
#include "ops/vsi_nn_op_unstack.h"
#include "ops/vsi_nn_op_pre_process_rgb.h"
#include "ops/vsi_nn_op_pre_process.h"
#include "ops/vsi_nn_op_addn.h"
#include "ops/vsi_nn_op_softmax_internal.h"
#include "ops/vsi_nn_op_pre_process_yuv420.h"
#include "ops/vsi_nn_op_pre_process_yuv444.h"
#include "ops/vsi_nn_op_pre_process_nv12.h"
#include "ops/vsi_nn_op_extra_ending.h"
#include "ops/vsi_nn_op_gather.h"
#include "ops/vsi_nn_op_scatter_nd.h"
#include "ops/vsi_nn_op_tile.h"
#include "ops/vsi_nn_op_grouped_conv2d.h"
#include "ops/vsi_nn_op_topk.h"
#include "ops/vsi_nn_op_pre_process_bgra.h"
#include "ops/vsi_nn_op_logical_not.h"
#include "ops/vsi_nn_op_sin.h"
#include "ops/vsi_nn_op_log.h"
#include "ops/vsi_nn_op_argmin.h"
#include "ops/vsi_nn_op_roi_align.h"
#include "ops/vsi_nn_op_heatmap_max_keypoint.h"
#include "ops/vsi_nn_op_axis_aligned_bbox_transform.h"
#include "ops/vsi_nn_op_box_with_nms_limit.h"
#include "ops/vsi_nn_op_generate_proposals.h"
#include "ops/vsi_nn_op_detection_postprocess.h"
#include "ops/vsi_nn_op_random_multinomial.h"
#include "ops/vsi_nn_op_log_softmax.h"
#include "ops/vsi_nn_op_relu_keras.h"
#include "ops/vsi_nn_op_relu_keras_internal.h"
#include "ops/vsi_nn_op_reducesum_internal.h"
#include "ops/vsi_nn_op_reducemax_internal.h"
#include "ops/vsi_nn_op_reducemin_internal.h"
#include "ops/vsi_nn_op_gru_ovxlib.h"
#include "ops/vsi_nn_op_grucell_ovxlib.h"
#include "ops/vsi_nn_op_embedding_lookup.h"
#include "ops/vsi_nn_op_reduceprod_internal.h"
#include "ops/vsi_nn_op_reduceall_internal.h"
#include "ops/vsi_nn_op_reduceany_internal.h"
#include "ops/vsi_nn_op_unidirectional_sequence_rnn.h"
#include "ops/vsi_nn_op_quantized_16bit_lstm.h"
#include "ops/vsi_nn_op_bidirectional_sequence_rnn.h"
#include "ops/vsi_nn_op_bidirectional_sequence_lstm.h"
#include "ops/vsi_nn_op_resize_internal.h"
#include "ops/vsi_nn_op_resize_nearest_internal.h"
#include "ops/vsi_nn_op_variable.h"
#include "ops/vsi_nn_op_rnncell_ovxlib.h"
#include "ops/vsi_nn_op_l2_normalize.h"
#include "ops/vsi_nn_op_dataconvert.h"
#include "ops/vsi_nn_op_swish.h"
#include "ops/vsi_nn_op_cast.h"
#include "ops/vsi_nn_op_depthwise_conv1d.h"
#include "ops/vsi_nn_op_grucell_activation_internal.h"
#include "ops/vsi_nn_op_grucell_activation_internal_sma.h"
#include "ops/vsi_nn_op_linear.h"
#include "ops/vsi_nn_op_batchnorm_single.h"
#include "ops/vsi_nn_op_moments.h"
#include "ops/vsi_nn_op_squeeze.h"
#include "ops/vsi_nn_op_expand_broadcast.h"
#include "ops/vsi_nn_op_deconvolution1d.h"
#include "ops/vsi_nn_op_interp.h"
#include "ops/vsi_nn_op_resize_1d.h"
#include "ops/vsi_nn_op_resize_1d_bilinear_internal.h"
#include "ops/vsi_nn_op_resize_1d_nearest_internal.h"
#include "ops/vsi_nn_op_upsamplescale.h"
#include "ops/vsi_nn_op_groupnormalize.h"
#include "ops/vsi_nn_op_sequence_mask.h"
#include "ops/vsi_nn_op_repeat.h"
#include "ops/vsi_nn_op_one_hot.h"
#include "ops/vsi_nn_op_nms.h"
#include "ops/vsi_nn_op_grouped_conv1d.h"
#include "ops/vsi_nn_op_scatter_nd_update.h"
#include "ops/vsi_nn_op_gelu.h"
#include "ops/vsi_nn_op_conv2d_lstm.h"
#include "ops/vsi_nn_op_conv2d_lstm_cell.h"
#include "ops/vsi_nn_op_gru.h"
#include "ops/vsi_nn_op_grucell.h"
#include "ops/vsi_nn_op_grucell_activation.h"
/* custom node head define define */
#include "custom/vsi_nn_custom_node_type.h"

#if defined(__cplusplus)
extern "C"{
#endif

/** Operation attributes */
typedef union _vsi_nn_nn_param
{
    struct
    {
        vsi_nn_conv2d_param         conv2d;
        vsi_nn_pool_param           pool;
    };
    vsi_nn_fcl_param                fcl;
    vsi_nn_activation_param         activation;
    vsi_nn_lrn_param                lrn;
    vsi_nn_concat_param             concat;
    vsi_nn_split_param              split;
    vsi_nn_roi_pool_param           roi_pool;
    vsi_nn_batch_norm_param         batch_norm;
    vsi_nn_multiply_param           multiply;
    vsi_nn_proposal_param           proposal;
    vsi_nn_deconv_param             deconv;
    vsi_nn_reshape_param            reshape;
    vsi_nn_permute_param            permute;
    vsi_nn_upsample_param           upsample;
    vsi_nn_resize_param             resize;
    vsi_nn_lstm_param               lstm;
    vsi_nn_reorg_param              reorg;
    vsi_nn_l2normalizescale_param   l2normalizescale;
    vsi_nn_crop_param               crop;
    vsi_nn_relun_param              relun;
    vsi_nn_divide_param             divide;
    vsi_nn_tanh_param               tanh;
    vsi_nn_dropout_param            dropout;
    vsi_nn_shufflechannel_param     shufflechannel;
    vsi_nn_prelu_param              prelu;
    vsi_nn_elu_param                elu;
    vsi_nn_reverse_param            reverse;
    vsi_nn_space2depth_param        space2depth;
    vsi_nn_space2depth_internal_param space2depth_internal;
    vsi_nn_depth2space_param        depth2space;
    vsi_nn_depth2space_internal_param depth2space_internal;
    vsi_nn_maximum_param            maximum;
    vsi_nn_scale_param              scale;
    vsi_nn_slice_param              slice;
    vsi_nn_space2batch_param        space2batch;
    vsi_nn_batch2space_param        batch2space;
    vsi_nn_pad_param                pad;
    vsi_nn_imageprocess_param       imageprocess;
    vsi_nn_matrixmul_param          matrixmul;
    vsi_nn_lstmunit_param           lstmunit;
    vsi_nn_layernormalize_param     layernorm;
    vsi_nn_reduce_param             reduce;
    vsi_nn_instancenormalize_param  instancenorm;
    vsi_nn_tensorstackconcat_param  tensorstackconcat;
    vsi_nn_softmax_param            softmax;
    vsi_nn_strided_slice_param      strided_slice;
    vsi_nn_signalframe_param        signalframe;
    vsi_nn_svdf_param               svdf;
    vsi_nn_conv1d_param             conv1d;
    vsi_nn_nbg_param                nbg;
    vsi_nn_concatshift_param        concatshift;
    vsi_nn_relational_ops_param     relational_ops;
    vsi_nn_pow_param                pow;
    vsi_nn_floordiv_param           floordiv;
    vsi_nn_spatial_transformer_param spatial_transformer;
    vsi_nn_logical_ops_param        logical_ops;
    vsi_nn_select_param             select;
    vsi_nn_lstmunit_activation_param lstmunit_activation;
    vsi_nn_lstmunit_ovxlib_param    lstmunit_ovxlib;
    vsi_nn_tensor_add_mean_stddev_norm_param tensor_add_mean_stddev_norm;
    vsi_nn_lstm_ovxlib_param        lstm_ovxlib;
    vsi_nn_lsh_projection_param     lsh_projection;
    vsi_nn_rnn_param                rnn;
    vsi_nn_stack_param              stack;
    vsi_nn_floor_param              floor;
    vsi_nn_neg_param                neg;
    vsi_nn_exp_param                exp;
    vsi_nn_clip_param               clip;
    vsi_nn_pre_process_tensor_param pre_process_tensor;
    vsi_nn_post_process_param       post_process;
    vsi_nn_pre_process_gray_param   pre_process_gray;
    vsi_nn_unstack_param            unstack;
    vsi_nn_pre_process_rgb_param    pre_process_rgb;
    vsi_nn_pre_process_param        pre_process;
    vsi_nn_addn_param               addn;
    vsi_nn_softmax_internal_param   softmax_internal;
    vsi_nn_pre_process_yuv420_param pre_process_yuv420;
    vsi_nn_pre_process_yuv444_param pre_process_yuv444;
    vsi_nn_pre_process_nv12_param   pre_process_nv12;
    vsi_nn_extra_ending_param       extra_ending;
    vsi_nn_gather_param             gather;
    vsi_nn_scatter_nd_param         scatter_nd;
    vsi_nn_tile_param               tile;
    vsi_nn_grouped_conv2d_param     grouped_conv2d;
    vsi_nn_topk_param               topk;
    vsi_nn_pre_process_bgra_param   pre_process_bgra;
    vsi_nn_logical_not_param        logical_not;
    vsi_nn_argmax_param             argmax;
    vsi_nn_sin_param                sin;
    vsi_nn_log_param                log;
    vsi_nn_argmin_param             argmin;
    vsi_nn_roi_align_param          roi_align;
    vsi_nn_heatmap_max_keypoint_param heatmap_max_keypoint;
    vsi_nn_axis_aligned_bbox_transform_param axis_aligned_bbox_transform;
    vsi_nn_box_with_nms_limit_param box_with_nms_limit;
    vsi_nn_generate_proposals_param generate_proposals;
    vsi_nn_detection_postprocess_param detection_postprocess;
    vsi_nn_random_multinomial_param random_multinomial;
    vsi_nn_log_softmax_param        log_softmax;
    vsi_nn_relu_keras_param         relu_keras;
    vsi_nn_relu_keras_internal_param relu_keras_internal;
    vsi_nn_reducesum_internal_param reducesum_internal;
    vsi_nn_reducemax_internal_param reducemax_internal;
    vsi_nn_reducemin_internal_param reducemin_internal;
    vsi_nn_gru_ovxlib_param         gru_ovxlib;
    vsi_nn_grucell_ovxlib_param     grucell_ovxlib;
    vsi_nn_embedding_lookup_param   embedding_lookup;
    vsi_nn_reduceprod_internal_param reduceprod_internal;
    vsi_nn_reduceall_internal_param reduceall_internal;
    vsi_nn_reduceany_internal_param reduceany_internal;
    vsi_nn_unidirectional_sequence_rnn_param unidirectional_sequence_rnn;
    vsi_nn_quantized_16bit_lstm_param quantized_16bit_lstm;
    vsi_nn_bidirectional_sequence_rnn_param bidirectional_sequence_rnn;
    vsi_nn_bidirectional_sequence_lstm_param bidirectional_sequence_lstm;
    vsi_nn_resize_internal_param    resize_internal;
    vsi_nn_resize_nearest_internal_param resize_nearest_internal;
    vsi_nn_variable_param variable;
    vsi_nn_rnncell_ovxlib_param     rnncell_ovxlib;
    vsi_nn_l2_normalize_param       l2_normalize;
    vsi_nn_depthwise_conv1d_param   depthwise_conv1d;
    vsi_nn_cast_param               cast;
    vsi_nn_swish_param              swish;
    vsi_nn_dataconvert_param        dataconvert;
    vsi_nn_grucell_activation_internal_param grucell_activation_internal;
    vsi_nn_grucell_activation_internal_sma_param grucell_activation_internal_sma;
    vsi_nn_linear_param             linear;
    vsi_nn_batchnorm_single_param   batchnorm_single;
    vsi_nn_moments_param            moments;
    vsi_nn_squeeze_param            squeeze;
    vsi_nn_expand_broadcast_param   expand_broadcast;
    vsi_nn_deconvolution1d_param    deconvolution1d;
    vsi_nn_interp_param             interp;
    vsi_nn_resize_1d_param          resize_1d;
    vsi_nn_resize_1d_bilinear_internal_param resize_1d_bilinear_internal;
    vsi_nn_resize_1d_nearest_internal_param resize_1d_nearest_internal;
    vsi_nn_upsamplescale_param      upsamplescale;
    vsi_nn_groupnormalize_param     groupnorm;
    vsi_nn_sequence_mask_param      sequence_mask;
    vsi_nn_repeat_param             repeat;
    vsi_nn_one_hot_param            one_hot;
    vsi_nn_nms_param                nms;
    vsi_nn_grouped_conv1d_param     grouped_conv1d;
    vsi_nn_scatter_nd_update_param  scatter_nd_update;
    vsi_nn_gelu_param               gelu;
    vsi_nn_conv2d_lstm_param        conv2d_lstm;
    vsi_nn_conv2d_lstm_cell_param   conv2d_lstm_cell;
    vsi_nn_gru_param                gru;
    vsi_nn_grucell_param            grucell;
    vsi_nn_grucell_activation_param grucell_activation;
    uint8_t                         client_param[128];

    /* custom node data struct define */
#define DEF_NODE_TYPE( NAME ) vsi_nn_##NAME##_param NAME;
    #include "custom/custom_node_type.def"
#undef DEF_NODE_TYPE
} vsi_nn_nn_param_t;

/**
 * Number 576 is the size of `vsi_nn_nn_param_t` from V1.1.2
 * We this check to avoid application binary interface(ABI) compatibility issue.
 */
_compiler_assert( sizeof(vsi_nn_nn_param_t) <= 576, vsi_nn_node_type_h_potential_abi_compatibility_issue );

/** Node params for openvx attributes */
typedef struct _vsi_nn_vx_param
{
    vsi_enum   overflow_policy;
    vsi_enum   rounding_policy;
    vsi_enum   down_scale_size_rounding;
    vsi_bool   has_relu;
    uint32_t accumulator_bits;
    vsi_nn_platform_e platform;
} vsi_nn_vx_param_t;

#if defined(__cplusplus)
}
#endif

#endif
