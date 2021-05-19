/* 

 * Copyright (c) 2012-2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _VX_KHR_NN_INTERNAL_H_
#define _VX_KHR_NN_INTERNAL_H_

/*!
 * \file
 * \brief The Khronos Extension for Deep Convolutional Networks Functions.
 *
 * \defgroup group_cnn Extension: Deep Convolutional Networks API
 * \brief Convolutional Network Nodes.
 */

#define OPENVX_KHR_NN_INTERNAL   "vx_khr_nn_internal"

#include <VX/vx.h>


#ifdef  __cplusplus
extern "C" {
#endif

/*TODO: check it for OpenVX 1.2*/
//#if defined(OPENVX_CNN_1_0)
//#undef OPENVX_CNN_1_1
//#endif

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and pooling Layer Node.
* \details This function implement Convolutional Network Convolution and Activation(Relu) and pooling layer.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
* The dimension order is [width, height, #IFM, #batches]. \n
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.\n
* \param [in] pad_x [static] Number of elements added at each side in the x dimension of the input.
* \param [in] pad_y [static] Number of elements added at each side in the y dimension of the input. In fully connected layers this input is ignored.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function
* \param [in] pool_type [static] if neither max pooling nor average pooling, disable pooling function. (see <tt>\ref vx_convolutional_network_pooling_type_e</tt>).
* \param [in] pool_size_x [static] Size of the pooling region in the x dimension
* \param [in] pool_size_y [static] Size of the pooling region in the y dimension.
* \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingLayer(
    vx_graph                    graph, 
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad_x,
    vx_uint32                   pad_y,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_enum                     pool_type,
    vx_uint32                   pool_size_x,
    vx_uint32                   pool_size_y,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) Layer Node.
* \details This function implement Convolutional Network Convolution and Activation(Relu) layer.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.
* \param [in] pad_x [static] Number of elements added at each side in the x dimension of the input.
* \param [in] pad_y [static] Number of elements added at each side in the y dimension of the input. In fully connected layers this input is ignored.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function.
* \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/

VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluLayer(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad_x,
    vx_uint32                   pad_y,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Fully connected and Activation(Relu) Convolutional Network Layer Node.
* \details This function implement Fully connected and Activation(Relu) Convolutional Network layers.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. There two possible input layouts:
* 1. [#IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>.
* 2. [width, height, #IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>\n
* In both cases number of batches are optional and may be multidimensional.
* The second option is a special case to deal with convolution layer followed by fully connected.
* The dimension order is [#IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>. Note that batch may be multidimensional.
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.\n
* \param [in] pad [static] Number of elements added at each side in the input.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function.
* \param [out] outputs The output tensor data. Output dimension layout is [#OFM,#batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>, where #batches may be multidimensional.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxFullyConnectedReluLayer(
    vx_graph                    graph, 
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_tensor                   outputs
    );

/*! \brief Input parameter for convolutionReluPooling2
 * \ingroup group_cnn
 */
typedef struct _vx_nn_convolution_relu_pooling_params_t
{
    vx_size   dilation_x;                /*!< \brief  "inflate" the kernel by inserting zeros between the kernel elements in the x direction. 
                                              The value is the number of zeros to insert. */
    vx_size   dilation_y;                /*!< \brief  "inflate" the kernel by inserting zeros between the kernel elements in the y direction. 
                                              The value is the number of zeros to insert. */
    vx_uint32  pad_x_left;                /*!< \brief  Number of elements added at each side in the left of x dimension of the input. */
    vx_uint32  pad_x_right;               /*!< \brief  Number of elements added at each side in the right of x dimension of the input. */
    vx_uint32  pad_y_top;                 /*!< \brief  Number of elements added at each side in the top of y dimension of the input. */
    vx_uint32  pad_y_bottom;              /*!< \brief  Number of elements added at each side in the bottom of y dimension of the input. */
    vx_uint8   accumulator_bits;          /*!< \brief  Is the total number of bits used during intermediate accumulation. */
    vx_enum    overflow_policy;           /*!< \brief  A VX_TYPE_ENUM of the vx_convert_policy_e enumeration. */
    vx_enum    rounding_policy;           /*!< \brief  A VX_TYPE_ENUM of the vx_round_policy_e enumeration. */
    vx_enum    down_scale_size_rounding;  /*!< \brief  Rounding method for calculating output dimensions. See vx_convolutional_network_rounding_type_e */
    vx_bool    enable_relu;               /*!< \brief  Enable Relu layer function or not. */
    vx_enum    pool_type;                 /*!< \brief  neither max pooling nor average pooling, disable pooling function (see vx_convolutional_network_pooling_type_e). */
    vx_uint32  pool_size_x;               /*!< \brief  Size of the pooling region in the x dimension */
    vx_uint32  pool_size_y;               /*!< \brief  Size of the pooling region in the y dimension. */
    vx_enum    pad_mode;                  /*!< \brief  A VX_TYPE_ENUM of the <tt> \ref vx_pad_mode_e </tt> enumeration. */
    vx_scalar  pad_const;                 /*!< \brief  The order const value if setting pad mode to const, the const value is base value, not quantized value. */
} vx_nn_convolution_relu_pooling_params_t, * vx_nn_convolution_relu_pooling_params;

/*! \brief Extended input parameter for a convolutionReluPooling2 operation.
 * \ingroup group_cnn
 *\version 0.3
 */
typedef struct _vx_nn_convolution_relu_pooling_params_ext_t
{
    vx_nn_convolution_relu_pooling_params_t base;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt> */
    vx_uint32       stride_x;       /*!< \brief  skip x jump for down scale.  */
    vx_uint32       stride_y;       /*!< \brief  skip y jump for down scale.  */
} vx_nn_convolution_relu_pooling_params_ext_t, * vx_nn_convolution_relu_pooling_params_ext;

/*! \brief The 2nd version of extended input parameter for a convolutionReluPooling2 operation.
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_convolution_relu_pooling_params_ext2_t
{
    vx_nn_convolution_relu_pooling_params_ext_t ext;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params__ext_t</tt> */
    vx_int32        depth_multiplier; /*!< \brief  specifying the depthwise multiplier for depthwise convolution.  */
    vx_enum         src_rank_mode; /*!< \brief source rank mode A VX_TYPE_ENUM of the <tt> \ref vx_tensor_rank_type_e </tt> enumeration. */
    vx_enum         convert_dst_format;    /*!< \brief The convert target format. */
} vx_nn_convolution_relu_pooling_params_ext2_t, * vx_nn_convolution_relu_pooling_params_ext2;

#define MERGED_NODE_COUNT_MAX 4

typedef struct _vx_nn_convolution_relu_pooling_params_ext3_t
{
    vx_nn_convolution_relu_pooling_params_ext2_t ext2;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params__ext_t</tt> */
    vx_uint32       mergedNodeCount;
    vx_float32*     interScale; /*!< \brief  specifying the depthwise multiplier for depthwise convolution.  */
    vx_int32*       interZeroPoint;
    vx_enum*        interDataType;
} vx_nn_convolution_relu_pooling_params_ext3_t, * vx_nn_convolution_relu_pooling_params_ext3;

typedef struct _vx_nn_convolution_relu_pooling_params_ext4_t
{
    vx_nn_convolution_relu_pooling_params_ext3_t ext3;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params__ext_t</tt> */
    vx_uint32       poolingStrideX;
    vx_uint32       poolingStrideY;
    vx_uint32       poolingPadLeft;
    vx_uint32       poolingPadRight;
    vx_uint32       poolingPadTop;
    vx_uint32       poolingPadBottom;
} vx_nn_convolution_relu_pooling_params_ext4_t, * vx_nn_convolution_relu_pooling_params_ext4;

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and Pooling Layer Node, this fucntion match kronos NN Extension 1.2 verion.
 * \details This function implement Convolutional Network Convolution and Activation(Relu) and Pooling layer.
 *  For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] = saturate(round(\sum_{l} (\sum_{m,n} inputs[j-m,k-n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Convolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for index along the width dimension and y for index along the height dimension.\n
 * before the Convolution is done, a padding with zeros of the width and height input dimensions is performed.
 * Then down scale is done by picking the results according to a skip jump. The skip in the x and y is determined by the output size dimensions.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} = round(\frac{(width_{input} + paddingleft_x + paddingright_x - kernel_x - (kernel_x -1) * dilation_x)}{skip_x} + 1) \f$\n
 * and \n
 * \f$ height_{output} = round(\frac{(height + paddingtop_y + paddingbottom_y - kernel_y - (kernel_y -1) * dilation_y)}{skip_y} + 1) \f$\n 
 * where \f$width\f$ is the size of the input width dimension. \f$height\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height dimensions.
 * skip is calculated by the relation between input and output.
 * rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n  
 * \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference. 
 * \param [in] convolution_relu_pooling_params [static] Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params [static] Size in bytes of convolution_relu_pooling_params.
 * \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input. 
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingLayer2(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    const vx_nn_convolution_relu_pooling_params_t * convolution_relu_pooling_params,
    vx_size                     size_of_convolution_relu_pooling_params,
    vx_tensor                   outputs);

/*! \brief The optimization direvative for weights_biases_parameter create.
 * \ingroup group_cnn
 */
typedef struct _vx_weights_biases_parameter_optimizations_t {
    vx_int8  zrl;             /*!< \brief The zero run length. Set negtive value to disable*/
    vx_enum  outputFormat;    /*!< \brief The output format. */
    vx_int32 inputZeroPoint;  /*!< \brief  zero point of input. A 32 bit integer, in range [0, 255], Set zero value to disable */
} vx_weights_biases_parameter_optimizations_t;

typedef struct _vx_weights_biases_parameter_optimizations_ext_t {
    vx_int8  zrl;             /*!< \brief The zero run length. Set negtive value to disable*/
    vx_enum  outputFormat;    /*!< \brief The output format. */
    vx_int32 inputZeroPoint;  /*!< \brief  zero point of input. A 32 bit integer, in range [0, 255], Set zero value to disable */
    vx_uint32 num_of_input_dims; /*< \brief The input dimesion number*/
    vx_uint32 num_of_output_dims; /*!< \brief The output dimesion number*/
} vx_weights_biases_parameter_optimizations_ext_t;


typedef struct _vx_weights_biases_parameter_optimizations_ext2_t {
    vx_weights_biases_parameter_optimizations_ext_t ext;
    vx_float32 inputScale;
    vx_float32 outputScale;
    vx_enum    inputFormat;
    vx_int32 output_ZP_dw;        /*depthwise conv output ZP*/
    vx_float32 output_scale_dw;   /*depthwise conv output scale*/
    vx_int8  output_fpp_dw;       /*depthwise conv output fix-point*/
} vx_weights_biases_parameter_optimizations_ext2_t;

/*!
 * \brief Creates a reference to a vx_weights_biases_parameter opaque object.
 *
 * \param [in] layer_type                The network type of objects to hold. Types allowed are: 
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims               The dimention number of input & output image tensor.
 * \param [in] inputs_dims               The input tensor's dimension size.
 * \param [in] pad_x                     The number of elements subtracted at each side in the x dimension of the input.
 * \param [in] pad_y                     The number of elements subtracted at each side in the y dimension of the input.
 * \param [in] pooling_size_x            The size of the pooling region in the x dimension, 0 means no pooling operation.
 * \param [in] pooling_size_y            The size of the pooling region in the y dimension, 0 means no pooling operation.
 * \param [in] down_scale_size_rounding  A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
 * \param [in] convolution_outputs_dims  The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims         The output's dimension size after pooling operation.
 * \param [in] optimizations             A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] weights                   The weights tensor which need be compressed.
 * \param [in] biases                    The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL 
vxCreateWeightsBiasesParameterFromTensors(
    vx_enum layer_type,
    vx_uint32 num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 pad_x,
    vx_uint32 pad_y,
    vx_uint32 pooling_size_x,
    vx_uint32 pooling_size_y,
    vx_enum down_scale_size_rounding,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_tensor weights, 
    vx_tensor biases);

/*!
 * \brief Creates a reference to an opaque vx_weights_biases_parameter object.
 *
 * \param [in] layer_type                              The network type of objects to hold. Types allowed are: 
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims                             The dimention number of input & output image tensor.
 * \param [in] inputs_dims                             The input tensor's dimension size.
 * \param [in] convolution_outputs_dims                The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims                       The output's dimension size after pooling operation.
 * \param [in] output_format                           The output tensor element type.
 * \param [in] convolution_relu_pooling_params         The convolution_relu_pooling_params Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params The size in bytes of convolution_relu_pooling_params.
 * \param [in] optimizations                           A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] weights                                 The weights tensor which need be compressed.
 * \param [in] biases                                  The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL vxCreateWeightsBiasesParameterFromTensors2(
    vx_enum     layer_type,
    vx_uint32   num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_enum     output_format,
    const vx_nn_convolution_relu_pooling_params convolution_relu_pooling_params,
    vx_size size_of_convolution_relu_pooling_params,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_tensor   weights,
    vx_tensor   biases);

/*!
 * \brief Creates a reference to an opaque vx_weights_biases_parameter object.
 *
 * \param [in] layer_type                              The network type of objects to hold. Types allowed are: 
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] inputs_dims                             The input tensor's dimension size.
 * \param [in] convolution_outputs_dims                The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims                       The output's dimension size after pooling operation.
 * \param [in] convolution_relu_pooling_params         The convolution_relu_pooling_params Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params The size in bytes of convolution_relu_pooling_params.
 * \param [in] optimizations                           A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] size_of_optimizations                   The size in bytes of optimizations.
 * \param [in] weights                                 The weights tensor which need be compressed.
 * \param [in] biases                                  The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL vxCreateWeightsBiasesParameterFromTensors3(
    vx_enum     layer_type,
    vx_uint32 * inputs_dims,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    const vx_nn_convolution_relu_pooling_params convolution_relu_pooling_params,
    vx_size size_of_convolution_relu_pooling_params,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_size size_of_optimizations,
    vx_tensor   weights,
    vx_tensor   biases);

/*! \brief Releases the OpenVX object vx_weights_biases_parameter.
 * \param [in] weights_bias The pointer to the reference to the vx_weights_biases_parameter.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If weights_bias is not a <tt> vx_weights_biases_parameter</tt>.
 * \pre <tt>\ref vxCreateWeightsBiasesParameterFromTensors / vxCreateWeightsBiasesParameterFromTensors2/ vxCreateWeightsBiasesParameter / vxCreateWeightsBiasesParameterFromStream</tt>
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseWeightsBiasesParameter(vx_weights_biases_parameter *weights_bias);

/*!
 * \brief Creates a reference to an vx_weights_biases_parameter object.
 * \param [in] context                   The OpenVX context object.
 * \param [in] layer_type                The network type of objects to hold. Types allowed are: 
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims               The dimention number of input & output image tensor.
 * \param [in] inputs_dims               The input tensor's dimension size.
 * \param [in] pad_x                     The number of elements subtracted at each side in the x dimension of the input.
 * \param [in] pad_y                     The number of elements subtracted at each side in the y dimension of the input.
 * \param [in] pooling_size_x            The size of the pooling region in the x dimension, 0 means no pooling operation.
 * \param [in] pooling_size_y            The size of the pooling region in the y dimension, 0 means no pooling operation.
 * \param [in] down_scale_size_rounding  A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
 * \param [in] convolution_outputs_dims  The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims         The output's dimension size after pooling operation.
 * \param [in] weights_num_of_dims       The dimention number of weights tensor.
 * \param [in] weights_dims              The dimention size of weights tensor.
 * \param [in] weights_data_format       The format of weights tensor.
 * \param [in] weights_fixed_point_pos   The fixed point position when the weights element type is int16/int8, if 0 calculations are performed in integer math.
 * \param [in] biases_num_of_dims        The dimention number of biases tensor.
 * \param [in] biases_dims               The dimention size of biases tensor.
 * \param [in] biases_data_format        The format of biases tensor.
 * \param [in] biases_fixed_point_pos    The fixed point position when the biases element type is int16/int8, if 0 calculations are performed in integer math.
 * \param [in] raw_data_size             The data size of compressed data.
 *
 * \returns A weightsbiases reference without compressed kernel data <tt>vx_weights_biases_parameter</tt>. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL
vxCreateWeightsBiasesParameter(
    vx_context context,
    vx_enum layer_type,
    vx_uint32 num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 pad_x,
    vx_uint32 pad_y,
    vx_uint32 pooling_size_x,
    vx_uint32 pooling_size_y,
    vx_enum down_scale_size_rounding,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_uint32 weights_num_of_dims,
    vx_uint32 * weights_dims,
    vx_enum weights_data_format,
    vx_int8 weights_fixed_point_pos,
    vx_uint32 biases_num_of_dims,
    vx_uint32 * biases_dims,
    vx_enum biases_data_format,
    vx_int8 biases_fixed_point_pos,
    vx_uint32 raw_data_size
    );

/*! \brief Input parameters for a gru operation.
 * \ingroup group_cnn
 * \version 0.5
 */
typedef struct _vx_nn_gru_params_t
{
    vx_tensor reset2input_weights;                 /*!< \brief [static] Weight matrix for the reset gate with input. A 2-D tensor of type T, of shape [input_size, cell_size]. where "cell_size" corresponds to the number of cell units.*/
    vx_tensor update2input_weights;                /*!< \brief [static] Weight matrix for the update gate with input. A 2-D tensor of type T, of shape [input_size, cell_size]. */
    vx_tensor reset2recurrent_weights;             /*!< \brief [static] Weight matrix for the reset gate with recurrent(h_prev). A 2-D tensor of type T, of shape [cell_size, cell_size]. */
    vx_tensor update2recurrent_weights;            /*!< \brief [static] Weight matrix for the update gate with recurrent(h_prev). A 2-D tensor of type T, of shape [cell_size, cell_size]. */

    vx_tensor connection2input_weights;            /*!< \brief [static] Weight matrix for the cell connection gate with input. A 2-D tensor of type T, of shape [input_size, cell_size]. */
    vx_tensor connection2recurrent_weights;        /*!< \brief [static] Weight matrix for the cell connection gate with recurrent(h_prev). A 2-D tensor of type T, of shape [cell_size, cell_size]. */
    
    vx_tensor gate_input_bias;                     /*!< \brief [static] Bias vector for the reset and update gate for input. A 1-D tensor of type T, of shape [cell_size].*/
    vx_tensor gate_recurrent_bias;                 /*!< \brief [static] Bias vector for the reset and update gate for recurrent. A 1-D tensor of type T, of shape [cell_size].*/

    vx_tensor connection_bias;                     /*!< \brief [static] Bias vector for the cell connection gate. A 1-D tensor of type T, of shape [cell_size].*/
    
} vx_nn_gru_params_t;


/*! \brief [Graph] Creates a Long short-term memory unit (gru) Unit Networks Layer Node. not implement yet.
 * \details
 *  The implementation is based on:  http://arxiv.org/abs/1406.1078
 *    Computes the GRU cell forward propagation for 1 time step.
 *    This kernel op implements the following mathematical equations:
 *    Biases are initialized with:
 *    * `b_ru` - constant_initializer(1.0)
 *    * `b_c` - constant_initializer(0.0)
 *
 *    x_h_prev = [x, h_prev]
 *    [r_bar u_bar] = x_h_prev * w_ru + b_ru
 *    r = sigmoid(r_bar)
 *    u = sigmoid(u_bar)
 *    h_prevr = h_prev x r
 *    x_h_prevr = [x h_prevr]
 *    c_bar = x_h_prevr * w_c + b_c
 *    c = tanh(c_bar)
 *    h = (1-u) x c + u x h_prev
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "batch_size" corresponds to the batching dimension, and "input_size"
 *                    is the size of the input.
 * \param [in] h_prev A 2-D tensor of type T, of shape [cell_size, batch_size].
 * \param [in] gru_params gru paraments <tt>\ref vx_nn_gru_params_t </tt>.
 * \param [in] size_of_gru_params [static] The size of the gru_params.
 * \param [out] output A 2-D tensor of type T, of shape [cell_size, batch_size]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.5
 */
VX_API_ENTRY vx_node VX_API_CALL vxGRUUnitLayer(
    vx_graph graph,
    vx_tensor input,
    vx_tensor h_prev,
    const vx_nn_gru_params_t * gru_params,
    vx_size size_of_gru_params,
    vx_tensor output);

/*! \brief [Graph] Creates a Long short-term memory layer (gru) Networks Layer Node. not implement yet.
 * \details
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input A 3-D tensor of type T, of shape [input_size, batch_size, time_step], where
 *                    "input_size" corresponds to the size of the input, and "batch_size"
 *                    is the batching dimension, time_step means time length actually used by the input.
 * \param [in] h_prev optional, A 2-D tensor of type T, of shape [cell_size, batch_size], where
 *                    "input_size" corresponds to the size of the input, and "batch_size"
 *                    is the batching dimension.
 * \param [in] vx_nn_gru_params gru paraments <tt>\ref vx_nn_gru_params_t </tt>.
 * \param [in] size_of_gru_layer_params [static] The size of the vx_nn_gru_params.
 * \param [out] output A 2-D tensor of type T, of shape [cell_size, batch_size]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.5
 */
VX_API_ENTRY vx_node VX_API_CALL vxGRULayer(
    vx_graph graph, 
    vx_tensor input,
    vx_tensor h_prev,
    const vx_nn_gru_params_t * gru_layer_params,
    vx_size size_of_gru_layer_params,
    vx_tensor output
    );


/*! \brief Input parameters for a convolution lstm operation.
 * \ingroup group_cnn
 * \version 0.5
 */
typedef struct _vx_nn_convlstm_params_t
{
    vx_tensor input2input_weight;                  /*!< \brief Optional A 2-D tensor of type T, of shape [num_units, input_size]. where "num_units" corresponds to the number of cell units.*/
    vx_tensor input2forget_weight;                 /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    vx_tensor input2cell_weight;                   /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    vx_tensor input2output_weight;                 /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    
    vx_tensor recurrent2input_weight;              /*!< \brief Optional A 2-D tensor of type T, of shape [num_units, output_size]. where "output_size" corresponds to either the number of cell units (i.e., "num_units"), or the second dimension of the "projection_weights", if defined.*/
    vx_tensor recurrent2forget_weight;             /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    vx_tensor recurrent2cell_weight;               /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    vx_tensor recurrent2output_weight;             /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    
    vx_tensor input_gate_bias;                     /*!< \brief Optional A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor forget_gate_bias;                    /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor cell_bias;                           /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor output_gate_bias;                    /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
 
    vx_tensor activation;                          /*!< \brief Optional. An ActivationFunctionType indicating the activation function. If "NONE" is specified then it results in a linear activation.If "NONE" is specified then it results in a linear activation.*/
 
    vx_float32 forget_bias;                        /*!< \brief  Float32[static] A bias for the forget gate. If set to 0.0f(by default) then bias is ignored.*/
    vx_bool skip_connection;                       /*< \brief  If set to `vx_true_e`, concatenate the input to the output of the conv LSTM. Default: `vx_false_e`.*/

} vx_nn_convlstm_params_t;

/*! \brief input parameters for a convolution lstm layer operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_convlstm_layer_params_t
{
    vx_nn_convlstm_params_t convlstm_param;              /*!< \brief  convolution lstm input param <tt>\ref vx_nn_convlstm_params_t</tt>.*/
    vx_enum             convlstm_layer_type;         /*!< \brief  convolution lstm layer type.*/
} vx_nn_convlstm_layer_params_t;


/*! \brief [Graph] Creates a Convolution Long short-term memory unit (ConvLSTM) Unit Networks Layer Node. not implement yet.
 * \details
 *
 * https://arxiv.org/pdf/1506.04214v1.pdf
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "batch_size" corresponds to the batching dimension, and "input_size"
 *                    is the size of the input.
 * \param [in] output_state_in A 2-D tensor of type T, of shape [output_size, batch_size].
 * \param [in] cell_state_in A 2-D tensor of type T, of shape [num_units, batch_size].
 * \param [in] convlstm_params LSTM paraments <tt>\ref vx_nn_convlstm_params_t </tt>.
 * \param [in] size_of_convlstm_params [static] The size of the convlstm_params.
 * \param [out] scratch A 3-D tensor of type T, of shape [num_cell, 4, batch_size].
 * \param [out] output_state_out A 2-D tensor of type T, of shape [output_size, batch_size].
 * \param [out] cell_state_out A 2-D tensor of type T, of shape [num_units, batch_size].
 * \param [out] output A 2-D tensor of type T, of shape [output_size, batch_size]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.5
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvLSTMUnitLayer(
    vx_graph graph,
    vx_tensor input,
    vx_tensor output_state_in,
    vx_tensor cell_state_in,
    const vx_nn_convlstm_params_t * convlstm_params,
    vx_size size_of_convlstm_params,
    vx_tensor output_state_out,
    vx_tensor cell_state_out,
    vx_tensor output);

/*! \brief [Graph] Creates a Long short-term memory layer (LSTM) Networks Layer Node. not implement yet.
 * \details
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input A 3-D tensor of type T, of shape [input_size, batch_size, time_step], where
 *                    "input_size" corresponds to the size of the input, and "batch_size"
 *                    is the batching dimension, time_step means time length actually used by the input.
 * \param [in] static_input optional, A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "input_size" corresponds to the size of the input, and "batch_size"
 *                    is the batching dimension.
 * \param [in] cont optional, A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "input_size" corresponds to the size of the input, and "batch_size"
 *                    is the batching dimension.
 * \param [in] convlstm_layer_params LSTM paraments <tt>\ref vx_nn_convlstm_layer_params_t </tt>.
 * \param [in] size_of_convlstm_layer_params [static] The size of the convlstm_layer_params.
 * \param [out] output A 2-D tensor of type T, of shape [output_size, batch_size]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.5
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvLSTMLayer(
    vx_graph graph, 
    vx_tensor input,
    vx_tensor static_input,
    vx_tensor cont,
    const vx_nn_convlstm_layer_params_t * convlstm_layer_params,
    vx_size size_of_convlstm_layer_params,
    vx_tensor output
    );

/*! \brief [Graph] Creates a Convolutional Network Pooling Layer Node.
 * \details Pooling is done on the first 2 dimensions or the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for the first dimension and y for the second.\n
 * Pooling operation is a function operation over a rectangle size and then a nearest neighbour down scale.
 * Here we use pool_size_x and pool_size_y to specify the rectangle size on which the operation
 * is performed. \n
 * before the operation is done (average or maximum value). the data is padded in the first 2D with zeros.
 * The down scale is done by picking the results according to a skip jump. The skip in the x and y dimension is determined by the output size dimensions.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional.Dimension layout is [width, height, #IFM, #batches].
* See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt> 
* \param [in] pool_type [static] Either max pooling or average pooling (see <tt>\ref vx_convolutional_network_pooling_type_e</tt>).
* \param [in] pool_size_x [static] Size of the pooling region in the x dimension
* \param [in] pool_size_y [static] Size of the pooling region in the y dimension. 
* \param [in] pool_pad_x [static] Padding size in the x dimension. 
* \param [in] pool_pad_y [static] Padding size in the y dimension.
* \param [in] rounding [static] The rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxPoolingLayer(vx_graph graph, vx_tensor inputs, vx_enum pooling_type,
        vx_size pooling_size_x,
        vx_size pooling_size_y,
        vx_size pooling_padding_x,
        vx_size pooling_padding_y,
        vx_enum rounding, 
        vx_tensor outputs);

/*! \brief [Graph] Creates a Convolutional Network Softmax Layer Node.
 * \details  the softmax function, is a generalization of the logistic function that "squashes" a K-dimensional vector \f$ z \f$ of arbitrary real values to a K-dimensional vector
 * \f$ \sigma(z) \f$ of real values in the range (0, 1) that add up to 1. The function is given by:
 * \f$ \sigma(z) = \frac{\exp^z}{\sum_i \exp^{z_i}} \f$
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor,  with the number of dimensions according to the following scheme. 
 * In case IFM dimension is 1. Softmax is be calculated on that dimension.
 * In case IFM dimension is 2. Softmax is be calculated on the first dimension. The second dimension is batching.
 * In case IFM dimension is 3. Dimensions are [Width, Height, Classes]. And Softmax is calculated on the third dimension.
 * In case IFM dimension is 4. Dimensions are [Width, Height, Classes, batching]. Softmax is calculated on the third dimension.
 * Regarding the layout specification, see <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>.
 * \param [out] outputs The output tensor. Output will have the same number of dimensions as input. Output tensor data type must be same as the inputs.
 * \ingroup group_cnn
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxSoftmaxLayer(vx_graph graph, vx_tensor inputs, vx_tensor outputs);

/* vxCopyTensorPatchForNN11 is for back compatibility with spec 1.1, which is used in nn*/
VX_API_ENTRY vx_status VX_API_CALL vxCopyTensorPatchForNN11(
    vx_tensor tensor,
    vx_tensor_view view,
    vx_tensor_addressing user_addr,
    void *user_ptr,
    vx_enum usage,
    vx_enum user_mem_type
    );

/* vxCreateTensorForNN11 is for back compatibility with spec 1.1, which is used in nn*/
VX_API_ENTRY vx_tensor VX_API_CALL
vxCreateTensorForNN11(
    vx_context context,
    vx_uint32 num_of_dims,
    vx_uint32 *sizes,
    vx_enum data_format,
    vx_int8 fixed_point_pos
    );

/*! \brief [Graph] Creates a Convolutional Network Normalization Layer Node.
* \details Normalizing over local input regions. Each input value is divided by \f$ (1+\frac{\alpha}{n}\sum_i x^2_i)^\beta \f$ , where n is the number of elements to normalize across.
* and the sum is taken over the region centred at that value (zero padding is added where necessary).
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional.Dimension layout is [width, height, IFM, #batches].
* See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
* \param [in] type [static] Either same map or across maps (see vx_convolutional_network_norm_type_e).
* \param [in] norm_size [static] Number of elements to normalize across.
* \param [in] alpha [static] Alpha parameter in the normalization equation.
* \param [in] beta  [static ] Beta parameter in the normalization equation.
* \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
* \ingroup group_cnn
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_enum type,
        vx_size normalization_size,
        vx_float32 alpha,
        vx_float32 beta,
        vx_tensor outputs);

/*! \brief [Graph] Creates a Reorgnization Layer Node.
 * \details Reorganize the layer. Picking up pixels from input tensor according to the rule \n
 * dimension 1: i * stride + (k / out_c) % stride \n
 * dimension 2: j * stride + (k / out_c) / stride \n
 * dimension 3: k % out_c  \n
 * out_c = input_c / (stride * stride), i is in range (0, input_w-1), j is in range (0, input_h-1), k is in range (0, input_c-1)
 * Output value is in order sequence.
 * \param [in] graph The reference to the parent graph.
 * \param [in] inputs The input tensor data to reorg.
 * \param [in] stride [static] Delta size of two pixels in each dimensions to do a reorg operation.
 * \param [out] outputs The output tensor data. Output will have different number of each dimensions as input.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxReorgLayer(
    vx_graph                    graph, 
    vx_tensor                   inputs,
    vx_uint32                   stride,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Convolutional Network L2Normalize Layer Node.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Dimension layout is [width, height, #IFM, #batches].
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
* \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
* \ingroup group_cnn
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxL2NormalizeLayer(vx_graph graph, vx_tensor inputs, vx_tensor outputs);

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and Pooling and Add Layer Node.
 * \details This function implement Convolutional Network Convolution and Activation(Relu) and Pooling and Add layer.
 *  For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] = saturate(round(\sum_{l} (\sum_{m,n} inputs[j-m,k-n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Convolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for index along the width dimension and y for index along the height dimension.\n
 * before the Convolution is done, a padding with zeros of the width and height input dimensions is performed.
 * Then down scale is done by picking the results according to a skip jump. The skip in the x and y is determined by the output size dimensions.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} = round(\frac{(width_{input} + paddingleft_x + paddingright_x - kernel_x - (kernel_x -1) * dilation_x)}{skip_x} + 1) \f$\n
 * and \n
 * \f$ height_{output} = round(\frac{(height + paddingtop_y + paddingbottom_y - kernel_y - (kernel_y -1) * dilation_y)}{skip_y} + 1) \f$\n 
 * where \f$width\f$ is the size of the input width dimension. \f$height\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height dimensions.
 * skip is calculated by the relation between input and output.
 * rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs_conv The input tensor data for convolution. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * \param [in] inputs_add The input tensor data for add. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n  
 * \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference. 
 * \param [in] convolution_relu_pooling_params [static] Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params [static] Size in bytes of convolution_relu_pooling_params.
 * \param [out] outputs_conv The convolution output tensor data. Output will have the same number and structure of dimensions as inputs_conv. 
 * \param [out] outputs_add The final add output tensor data. Output will have the same number and structure of dimensions as input. 
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingAddLayer2(
    vx_graph                    graph,
    vx_tensor                   inputs_conv,
    vx_tensor                   inputs_add,
    vx_weights_biases_parameter weights_biases,
    const vx_nn_convolution_relu_pooling_params_t * convolution_relu_pooling_params,
    vx_size                     size_of_convolution_relu_pooling_params,
    vx_tensor                   outputs_conv,
    vx_tensor                   outputs_add);

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and Pooling and Multiply Layer Node.
 * \details This function implement Convolutional Network Convolution and Activation(Relu) and Pooling and Multiply layer.
 *  For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] = saturate(round(\sum_{l} (\sum_{m,n} inputs[j-m,k-n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Convolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for index along the width dimension and y for index along the height dimension.\n
 * before the Convolution is done, a padding with zeros of the width and height input dimensions is performed.
 * Then down scale is done by picking the results according to a skip jump. The skip in the x and y is determined by the output size dimensions.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} = round(\frac{(width_{input} + paddingleft_x + paddingright_x - kernel_x - (kernel_x -1) * dilation_x)}{skip_x} + 1) \f$\n
 * and \n
 * \f$ height_{output} = round(\frac{(height + paddingtop_y + paddingbottom_y - kernel_y - (kernel_y -1) * dilation_y)}{skip_y} + 1) \f$\n 
 * where \f$width\f$ is the size of the input width dimension. \f$height\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height dimensions.
 * skip is calculated by the relation between input and output.
 * rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs_conv The input tensor data for convolution. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * \param [in] inputs_mul The input tensor data for mul. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n  
 * \param [in] scale A non-negative <tt>\ref VX_TYPE_FLOAT32</tt> multiplied to each product before overflow handling.
 * \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference. 
 * \param [in] convolution_relu_pooling_params [static] Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params [static] Size in bytes of convolution_relu_pooling_params.
 * \param [out] outputs_conv The convolution output tensor data. Output will have the same number and structure of dimensions as inputs_conv. 
 * \param [out] outputs_mul The final mul output tensor data. Output will have the same number and structure of dimensions as input. 
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingMultiplyLayer2(
    vx_graph                    graph,
    vx_tensor                   inputs_conv,
    vx_tensor                   inputs_mul,
    vx_float32                  input_scale,
    vx_weights_biases_parameter weights_biases,
    const vx_nn_convolution_relu_pooling_params_t * convolution_relu_pooling_params,
    vx_size                     size_of_convolution_relu_pooling_params,
    vx_tensor                   outputs_conv,
    vx_tensor                   outputs_mul);
/*! \brief [Graph] Performs LUT on element values in the input tensor data's.
 * \param [in] graph The handle to the graph.
 * \param [in] input input tensor data.
 * \param [in] InLut The look-up table of x value, of type <tt>\ref vx_lut</tt>.
 * \param [in] OutLut The look-up table of y value, of type <tt>\ref vx_lut</tt>.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data's.
 * \ingroup group_tensor
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorTableLookupLayer(
    vx_graph graph,
    vx_tensor input,
    vx_lut InLut,
    vx_lut OutLut,
    vx_tensor output);
#ifdef  __cplusplus
}
#endif


#endif
