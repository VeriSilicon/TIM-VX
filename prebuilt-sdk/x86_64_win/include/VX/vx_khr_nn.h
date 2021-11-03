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

#ifndef _VX_KHR_NN_H_
#define _VX_KHR_NN_H_

/*!
 * \file
 * \brief The Khronos Extension for Deep Convolutional Networks Functions.
 *
 * \defgroup group_cnn Extension: Deep Convolutional Networks API
 * \brief Convolutional Network Nodes.
 */

#define OPENVX_KHR_NN   "vx_khr_nn"

#include <VX/vx.h>
#include <VX/vx_khr_nn_internal.h>


#ifdef  __cplusplus
extern "C" {
#endif

/*TODO: check it for OpenVX 1.2*/
//#if defined(OPENVX_CNN_1_0)
//#undef OPENVX_CNN_1_1
//#endif

enum vx_context_attribute_internal_type_e
{
    VX_CONTEXT_DEVICE_COUNT_VIV                   = VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_CONTEXT) + 0x0,
};

enum vx_graph_attribute_internal_type_e
{
    /*! \brief Queries a graph for its device index (read-write. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_GRAPH_DEVICE_INDEX_VIV                     =  VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_GRAPH) + 0x0,
    /*! \brief Queries a graph for its weight data pre-loading size in vip sram (read-write. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_GRAPH_VIP_SRAM_PRE_LOAD                    =  VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_GRAPH) + 0x1,
    /*! \brief Queries a graph for its weight data pre-loading size in axi sram (read-write. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_GRAPH_AXI_SRAM_PRE_LOAD                    =  VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_GRAPH) + 0x2,
    /*! \brief Queries a graph for its running priority (read-write. Use a <tt>\ref vx_uint32</tt> parameter. */
    VX_GRAPH_PRIORITY_VALUE_VIV                   =  VX_ATTRIBUTE_BASE(VX_ID_VIVANTE, VX_TYPE_GRAPH) + 0x3,
};

/*! \brief Size Alignment of User Memory
 * \0x40   64Byte Align
 * \0x1000 4k Align
 */
#define VX_WRAP_USER_MEMORY_SIZE_ALIGNMENT (0x40)

/*! \brief OpenVX Version Compatibility set*/
#define VX_KHR_COMPATIBILITY (0x1)

/*==============================================================================
CONVOLUTIONAL_NETWORK structs and enums
=============================================================================*/
/*! \brief The Neural Network Extension Library Set
 * \ingroup group_cnn
 */
#define VX_LIBRARY_KHR_NN_EXTENSION (0x1) 

/*! \brief The list of Neural Network Extension Kernels.
 * \ingroup group_cnn
 */
enum vx_kernel_nn_ext_e {
    /*! \brief The Neural Network Extension convolution Kernel.
    * \see group_cnn
    */
    VX_KERNEL_CONVOLUTION_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x0,
    /*! \brief The Neural Network Extension fully connected Kernel.
    * \see group_cnn
    */
    VX_KERNEL_FULLY_CONNECTED_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x1,
    /*! \brief The Neural Network Extension pooling Kernel.
    * \see group_cnn
    */
    VX_KERNEL_POOLING_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x2,
    /*! \brief The Neural Network Extension softmax Kernel.
    * \see group_cnn
    */
    VX_KERNEL_SOFTMAX_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x3,
    /*! \brief The Neural Network Extension normalization Kernel.
    * \see group_cnn
    */
    VX_KERNEL_NORMALIZATION_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x4,
    /*! \brief The Neural Network Extension activation Kernel.
    * \see group_cnn
    */
    VX_KERNEL_ACTIVATION_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x5,
    /*! \brief The Neural Network POI Pooling Kernel.
    * \see group_cnn
    */
    VX_KERNEL_ROI_POOLING_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x6,
    /*! \brief The Neural Network Extension Deconvolution Kernel.
    * \see group_cnn
    */
    VX_KERNEL_DECONVOLUTION_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x7,
    /*! \brief The Neural Network Extension local response normalization Kernel (with bias).
    * \see group_cnn
    */
    VX_KERNEL_LOCAL_RESPONSE_NORMALIZATION_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_NN_EXTENSION) + 0x8,
};

/*! \brief NN extension type enums.
 * \ingroup group_cnn
 */
enum vx_nn_enum_e
{
    VX_ENUM_NN_ROUNDING_TYPE    = 0x1A,
    VX_ENUM_NN_POOLING_TYPE    = 0x1B,
    VX_ENUM_NN_NORMALIZATION_TYPE    = 0x1C,
    VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE    = 0x1D,
    /* 0x1E, 0x1F and 0x20 are reserved for VX_ENUM_CLASSIFIER_MODEL, VX_ENUM_IX_USE and VX_ENUM_SCALAR_OPERATION*/
    VX_ENUM_NN_LAYER_TYPE    = 0x21,
};

/*! \brief down scale rounding.
 * \details Due to different scheme of downscale size calculation in the various training frameworks. Implementation must support 2 rounding methods for down scale calculation.
 * The floor and the ceiling. In convolution and pooling functions.
 * Relevant when input size is even.
 * \ingroup group_cnn
 */
enum vx_nn_rounding_type_e
{
    /*! \brief floor rounding  */
    VX_NN_DS_SIZE_ROUNDING_FLOOR = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ROUNDING_TYPE) + 0x0,
    /*! \brief ceil rounding */
    VX_NN_DS_SIZE_ROUNDING_CEILING = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ROUNDING_TYPE) + 0x1
};


/*! \brief The Neural Network pooling type list.
 * \details kind of pooling done in pooling function
 * \ingroup group_cnn
 */
enum vx_nn_pooling_type_e
{
    /*! \brief max pooling*/
    VX_NN_POOLING_MAX = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_POOLING_TYPE) + 0x0,
    /*! \brief average pooling*/
    VX_NN_POOLING_AVG = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_POOLING_TYPE) + 0x1,
    /*! \brief l2 pooling*/
    VX_NN_POOLING_L2 = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_POOLING_TYPE) + 0x0,
    /*! \brief average pooling for android*/
    VX_NN_POOLING_AVG_ANDROID = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_POOLING_TYPE) + 0x1,
};


/*! \brief The Neural Network normalization type list.
 * \ingroup group_cnn
 */
enum vx_nn_norm_type_e
{
    /*! \brief normalization is done on same IFM*/
    VX_NN_NORMALIZATION_SAME_MAP = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_NORMALIZATION_TYPE) + 0x0,
    /*! \brief Normalization is done across different IFMs*/
    VX_NN_NORMALIZATION_ACROSS_MAPS = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_NORMALIZATION_TYPE) + 0x1,
};


/*! \brief The Neural Network activation functions list.
 * \details
 * <table>
 * <tr><td> <B>Function name </B> <td> <B>Mathematical definition</B> <td> <B>Parameters</B> <td> <B>Parameters type</B>
 * <tr><td>logistic <td> \f$f(x)=1/(1+e^{-x}) \f$  <td>  <td>
 * <tr><td>hyperbolic tangent <td> \f$f(x)=a\cdot tanh(b\cdot x) \f$  <td> a,b  <td> VX_FLOAT32
 * <tr><td>relu <td> \f$f(x)=max(0,x)\f$  <td>  <td>
 * <tr><td>bounded relu <td> \f$f(x)=min(a,max(0,x)) \f$  <td> a  <td> VX_FLOAT32
 * <tr><td>soft relu <td> \f$f(x)=log(1+e^{x}) \f$  <td>  <td>
 * <tr><td>abs <td> \f$f(x)=\mid x\mid \f$  <td>  <td>
 * <tr><td>square <td> \f$f(x)= x^2 \f$  <td>  <td>
 * <tr><td>square root <td> \f$f(x)=\sqrt{x} \f$  <td>  <td>
 * <tr><td>linear <td> \f$f(x)=ax+b \f$  <td>  a,b  <td> VX_FLOAT32
 * </table>
 * \ingroup group_cnn
 */
enum vx_nn_activation_function_e
{
    VX_NN_ACTIVATION_LOGISTIC = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x0,
    VX_NN_ACTIVATION_HYPERBOLIC_TAN = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x1,
    VX_NN_ACTIVATION_RELU = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x2,
    VX_NN_ACTIVATION_BRELU = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x3,
    VX_NN_ACTIVATION_SOFTRELU = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x4,
    VX_NN_ACTIVATION_ABS = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x5,
    VX_NN_ACTIVATION_SQUARE = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x6,
    VX_NN_ACTIVATION_SQRT = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x7,
    VX_NN_ACTIVATION_LINEAR = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x8,
    VX_NN_ACTIVATION_LEAKYRELU = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x0,
    VX_NN_ACTIVATION_RELU6 = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x1,
    VX_NN_ACTIVATION_RELU1 = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x2,
    VX_NN_ACTIVATION_RSQRT = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x3,
    VX_NN_ACTIVATION_LEAKYRELU_MAX_POOLING = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x4,
    VX_NN_ACTIVATION_SWISH = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x5,
    VX_NN_ACTIVATION_HSWISH = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x6,
    VX_NN_ACTIVATION_NONE = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_NN_ACTIVATION_FUNCTION_TYPE) + 0x7,
};

/*! \brief  The Convolutional network type 
 * \ingroup group_cnn
 */
enum vx_nn_layer_type_e
{
    /*! \brief convolution layer */
    VX_NN_CONVOLUTION_LAYER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_LAYER_TYPE) + 0x0,
    /*! \brief fully connected layer */
    VX_NN_FULLYCONNECTED_LAYER = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NN_LAYER_TYPE) + 0x1,
};

/*! \brief The pad mode list.
 * \ingroup group_cnn
 * \version 0.3
 */
enum vx_pad_mode_e {
    /*! \brief For nodes that support this behavior, a constant value is
     * \e filled-in when accessing padding pixels.
     * eg. [1,2,3,4]->C,C,[1,2,3,4]C,C
     */
    VX_PAD_CONSTANT = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_BORDER) + 0x0,

    /*! \brief For nodes that support this behavior, a relicateion of the nearest
     * edge pixels value is given for padding pixels.
     * eg. [1,2,3,4]->1,1,[1,2,3,4],4,4
     */
    VX_PAD_REPLICATE = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_BORDER) + 0x1,

    /*! \brief For nodes that support this behavior, a mirror of the nearest
     * edge pixels value is given for padding pixels. ege is duplicate.
     * eg. [1,2,3,4]->2,1,[1,2,3,4],4,3
     */
    VX_PAD_MIRROR_SYMMETRIC = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_BORDER) + 0x2,

    /*! \brief For nodes that support this behavior, a mirror of the nearest
     * edge pixels value is given for padding pixels. ege is not duplicate.
     * eg. [1,2,3,4]->3,2,[1,2,3,4],3,2
     */
    VX_PAD_MIRROR_REFLECT = VX_ENUM_BASE(VX_ID_VIVANTE, VX_ENUM_BORDER) + 0x3,
};

/*! \brief The Quantized format list.
 * \ingroup group_tensor
 * \version 0.3
 */
enum vx_quantized_format_e
{
    /*! \brief Non-quantized data. */
    VX_QUANT_NONE                   = 0x0,
    /*! \brief A quantization data type which specifies the fixed point position for whole tensor. */
    VX_QUANT_DYNAMIC_FIXED_POINT    = 0x1,
    /*! \brief A quantization data type which has scale value and zero point to match with TF and Android NN API for whole tensor. */
    VX_QUANT_AFFINE_SCALE           = 0x2,
    /*! \brief A quantization data type which has scale value and zero point to match with TF and Android NN API for per channel of tensor. */
    VX_QUANT_AFFINE_SCALE_PER_CHANNEL = 0x3,
};

/*! \brief The rank mode of tensor memory.
 * \ingroup group_tensor
 * \version 0.4
 */
enum vx_tensor_rank_type_e
{
    /*! \brief rank with weight,height,channel,batch */
    VX_TENSOR_RANK_WHCN = 0,

    /*! \brief  rank with channel,weight,height,batch */
    VX_TENSOR_RANK_CWHN,

    /*! \brief  rank with size, batch */
    VX_TENSOR_RANK_SN,
};

/*! \brief The precision of tensor.
 * \ingroup group_tensor
 * \version 0.4
 */
enum vx_tensor_precision_type_e
{
    /*! \brief auto adapter precision */
    VX_TENSOR_PRECISION_AUTO = 0,

    /*! \brief  high precision */
    VX_TENSOR_PRECISION_HIGH,
};

/*! \brief Specifies a static or dynamic tensor.
 * \ingroup group_tensor
 * \version 0.4
 */
enum vx_tensor_lifetime_type_e
{
    /*! \brief static tensor */
    VX_TENSOR_LIFE_TIME_STATIC = 0,

    /*! \brief  dynamic tensor */
    VX_TENSOR_LIFE_TIME_DYNAMIC,
};

/*==============================================================================
    TENSOR DATA FUNCTIONS
=============================================================================*/

/*! \brief Create  an opaque reference to a tensor view object.
 * \details Not guaranteed to exist until the <tt>vx_graph</tt> containing it has been verified.
 * \param [in] context The reference to the implementation context.
 * \param [in] view_array_start a vx_uint32 array of start values of the view.
 * \param [in] view_array_end a vx_uint32 array of end values of the view.
 * \param [in] numViewDimensions number of dimensions of view_array_start and view_array_end.
 * \return A tensor data view reference or zero when an error is encountered.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_tensor_view VX_API_CALL vxCreateTensorView(vx_context context, vx_uint32 *view_array_start, vx_uint32 * view_array_end, vx_uint8 numViewDimensions);

/*! \brief Releases a reference to a tensor data view object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] tensor_view The pointer to the tensor data view to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>vx_status_e</tt>.
* \ingroup group_tensor
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseTensorView(vx_tensor_view *tensor_view);

/*! \brief Create  an opaque reference to a tensor addressing object.
* \details Not guaranteed to exist until the <tt>vx_graph</tt> containing it has been verified.
* \param [in] context The reference to the implementation context.
* \param [in] addressing_array_dimension a vx_uint32 array of sLength of patch in all dimensions in elements.
* \param [in] addressing_array_stride a vx_uint32 arrayStride in all dimensions in bytes.
* \param [in] numViewDimensions number of dimensions of view_array_start and view_array_end.
* \return A tensor data view reference or zero when an error is encountered.
* \ingroup group_tensor
*/
VX_API_ENTRY vx_tensor_addressing VX_API_CALL vxCreateTensorAddressing(vx_context context, vx_uint32 *addressing_array_dimension, vx_uint32 * addressing_array_stride, vx_uint8 numViewDimensions);

/*! \brief Releases a reference to a tensor data addressing object.
* The object may not be garbage collected until its total reference count is zero.
* \param [in] tensor_addr The pointer to the tensor data addressing to release.
* \post After returning from this function the reference is zeroed.
* \return A <tt>vx_status_e</tt> enumeration.
* \retval VX_SUCCESS No errors.
* \retval VX_SUCCESS Success
* \retval * An error occurred. See <tt>vx_status_e</tt>.
* \ingroup group_tensor
*/
VX_API_ENTRY vx_status VX_API_CALL vxReleaseTensorAddressing(vx_tensor_addressing *tensor_addr);

/*! \brief Creates an array of tensors
 * \param [in] context      The reference to the overall Context.
 * \param [in] count        Number of Objects to create in the ObjectArray.
 * \param [in] tensor*     The tensors array that need add to the ObjectArray. 
 *
 * \returns An ObjectArray reference <tt>\ref vx_object_array</tt>. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>. Data objects are not initialized by this function.
 *
 * \ingroup group_object_array
 */
VX_API_ENTRY vx_object_array VX_API_CALL vxCreateTensorObjectArray(vx_context context, vx_uint32 count, vx_tensor* tensor);

typedef union _vx_tensor_quant_param
{
    struct 
    {
        vx_int8 fixed_point_pos; /*!< \brief Specifies the fixed point position when the input element type is int16/int8, if 0 calculations are performed in integer math */
    } dfp;

    struct 
    {
        vx_float32      scale;       /*!< \brief Scale vaule for the quantized value */
        vx_int32        zeroPoint;  /*!< \brief  A 32 bit integer, in range [0, 255] */
    } affine;

    struct 
    {
        vx_uint32       channelDim; /*!< \brief a 32 bit unsigned integer indicating channel dimension */
        vx_uint32       scaleCount; /*!< \brief the size of the scale array, must be equal to size[channelDim] */
        vx_float32 *    scales; /*!< \brief an array of positive 32 bit floating point value. The size of the scales array must be equal to size[channelDim] */
        vx_uint32       zeroPointCount; /*!< \brief the size of the zero point array, must be equal to 0 or size[channelDim] */
        vx_int32 *      zeroPoint;  /*!< \brief  A 32 bit integer, in range [0, 255] */
    } affinePerChannel;
}vx_tensor_quant_param;

/*! \brief Input parameter for createTensor2
 * \ingroup group_tensor
 * \version 0.3
 */
typedef struct _vx_tensor_create_params_t
{
    vx_uint32       num_of_dims; /*!< \brief The number of dimensions specified in *sizes*/
    vx_uint32 *     sizes;       /*!< \brief The pointer to an array of dimension */
    vx_enum         data_format; /*!< \brief Data format for the tensor */
    vx_enum         quant_format; /*!< \brief Quantized format <tt>\ref vx_quantized_format_e </tt>. */
    vx_tensor_quant_param quant_data;
} vx_tensor_create_params_t;


/*! \brief Creates an opaque reference to a tensor data buffer.
 * \details Not guaranteed to exist until the <tt>vx_graph</tt> containing it has been verified.
 * \param [in] context The reference to the implementation context.
 * \param [in] tensor_create_params A pointer to the tensor create parameter<tt>\ref vx_tensor_create_params_t</tt>
 * \param [in] size_of_create_params Byte size of the parameter structure
 * \return A tensor data reference or zero when an error is encountered.
 * \ingroup group_tensor
 * \version 0.3
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensor2(vx_context context, const vx_tensor_create_params_t* tensor_create_params, vx_size size_of_create_params);

/*! \brief Creates an opaque reference to a tensor data buffer with no direct
 * user access. This function allows setting the tensor data dimensions or data format.
 * \details Virtual data objects allow users to connect various nodes within a
 * graph via data references without access to that data, but they also permit the
 * implementation to take maximum advantage of possible optimizations. Use this
 * API to create a data reference to link two or more nodes together when the
 * intermediate data are not required to be accessed by outside entities. This API
 * in particular allows the user to define the tensor data format of the data without
 * requiring the exact dimensions. Virtual objects are scoped within the graph
 * they are declared a part of, and can't be shared outside of this scope.
 * \param [in] graph The reference to the parent graph.
 * \param [in] tensor_create_params A pointer to the tensor create parameter<tt>\ref vx_tensor_create_params_t</tt>
 * \param [in] size_of_create_params Byte size of the parameter structure
 * \return A tensor data reference or zero when an error is encountered.
 * \note Passing this reference to <tt>\ref vxCopyTensorPatch</tt> will return an error.
 * \ingroup group_tensor
 * \version 0.3
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateVirtualTensor2(vx_graph graph, const vx_tensor_create_params_t* tensor_create_params, vx_size size_of_create_params);

/*! \brief Swap tensor handle between two tensors which are created from handle.
 * \details These tensors must have the same attributes expect for tensor hanlde.
 * for better performance, must make sure the memory referenced by the tensor handle is flushed by using <tt>\ref vxFlushHandle</tt>.
 * \param [in] tensor0 The tensor whose handle will be changed to tensor1's.
 * \param [in] tensor1 The tensor whose handle will be changed to tensor0's.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE tensor is not a valid <tt>\ref vx_tensor</tt> reference.
 * reference.
 * \retval VX_ERROR_INVALID_REFERENCE The tensor0 and tensor1's attributes are not the same.
 * \ingroup group_tensor
 *\version 0.5
 */
VX_API_ENTRY vx_status VX_API_CALL vxSwapTensor(vx_tensor tensor0, vx_tensor tensor1);

/*! \brief Creates a reference to a tensor object that was externally allocated.
 * \param [in] context The reference to the implementation context.
 * \param [in] tensor_create_params The <tt>\ref vx_tensor_create_params_t</tt> that points to a parameter structure.
 * \param [in] size_of_create_params Size of parameter structure.
 * \param [in] addrs The tensor patch addressing structures that define the dimension and stride of pointers. See note below. 
 * \param [in] ptr The logical pointer of platform-defined references to tensor data.
 * \param [in] import_type <tt>\ref vx_memory_type_e</tt>. When giving <tt>\ref VX_MEMORY_TYPE_HOST</tt>
 * the \a ptr is assumed to be a HOST accessible pointer to memory.
 * \returns An tensor reference <tt>\ref vx_tensor</tt>. Any possible errors preventing a 
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * In order to release the image back to the application we should use <tt>\ref vxSwapTensorHandle</tt>.
 * 
 * \ingroup group_tensor
 *\version 0.4
 */
VX_API_ENTRY vx_tensor VX_API_CALL vxCreateTensorFromHandle2(
        vx_context context, const vx_tensor_create_params_t* tensor_create_params, vx_size size_of_create_params, const vx_tensor_addressing addrs, 
        void * const ptr, vx_enum import_type);
    
/*! \brief Flush the memory referenced by reference's handle when it is ready.
* \param [in] ref The reference(image or tensor) which created from handle.
* \return A <tt>\ref vx_status_e</tt> enumeration.;
* \retval VX_ERROR_INVALID_REFERENCE tensor is not a valid <tt>\ref vx_tensor</tt> <tt>\ref vx_image</tt>reference created from Handle.
*/
VX_API_ENTRY vx_status VX_API_CALL vxFlushHandle(vx_reference ref);


/*! \brief Return a new tensor referencing the same memory location but with different shape.
* \param [in] tensor The input tensor data to reshape.
* \param [in] num_of_dims Size of each dimension. If one component is special value -1,
* the size of that dimension is computed so that the total size remains the same as input tensor.
* If is is [-1], then flatten is performed which turns tensor into 1-D.
* \param [in] sizes The size of the container to which \a num_of_dims points.
* \return a vx_tensor that has shaped.
* \return VX_NULL if an error occurred.
* \ingroup group_tensor
*/
VX_API_ENTRY vx_tensor VX_API_CALL vxReshapeTensor(vx_tensor tensor, vx_int32* num_of_dims, vx_uint32 sizes);

/*! \brief Allows setting attributes on the tensor.
 * \param [in] tensor The reference to the tensor on which to set the attribute.
 * \param [in] attribute The attribute to set. Use a <tt>\ref vx_tensor_attribute_e</tt> enumeration.
 * \param [in] ptr The pointer to the location from which to read the value.
 * \param [in] size The size in bytes of the object pointed to by \a ptr.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If the tensor is not a <tt>\ref vx_tensor</tt>.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \ingroup group_tensor
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetTensorAttribute(vx_tensor tensor, vx_enum attribute, const void *ptr, vx_size size);


/*! \brief The type enumeration lists all NN extension types.
 * \ingroup group_cnn
 */
enum vx_nn_type_e {
    VX_TYPE_NN_CONVOLUTION_PARAMS     = 0x025,/*!< \brief A <tt>\ref vx_nn_convolution_params_t</tt>. */
    VX_TYPE_NN_DECONVOLUTION_PARAMS   = 0x026,/*!< \brief A <tt>\ref vx_nn_deconvolution_params_t</tt>. */
    VX_TYPE_NN_ROI_POOL_PARAMS        = 0x027,/*!< \brief A <tt>\ref vx_nn_roi_pool_params_t</tt>. */
};

/*! \brief Input parameters for a convolution operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_convolution_params_t
{
    vx_size padding_x;                 /*!< \brief Number of elements added at each side in the x dimension of the input. */
    vx_size padding_y;                 /*!< \brief Number of elements added at each side in the y dimension of the input. */
    vx_enum overflow_policy;         /*!< \brief A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration. */
    vx_enum rounding_policy;         /*!< \brief A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration. */
    vx_enum down_scale_size_rounding; /*!< \brief Rounding method for calculating output dimensions. See <tt>\ref vx_nn_rounding_type_e</tt> */
    vx_size dilation_x;            /*!< \brief "inflate" the kernel by inserting zeros between the kernel elements in the x direction. The value is the number of zeros to insert.*/
    vx_size dilation_y;            /*!< \brief "inflate" the kernel by inserting zeros between the kernel elements in the y direction. The value is the number of zeros to insert.*/
} vx_nn_convolution_params_t;

/*! \brief Extended input parameter structure for convolution layer
 * \ingroup group_cnn
 */
typedef struct _vx_nn_convolution_params_ext_t
{
    vx_nn_convolution_params_t khr;          /*!< \brief Khronos standard structure head */
    vx_size padding_x_right;                 /*!< \brief Number of elements added at each side in the right of x dimension of the input, 
                                               "padding_x" is for the left */
    vx_size padding_y_bottom;                /*!< \brief Number of elements added at each side in the bottom of y dimension of the input.
                                                "padding_y" is for the top */
    vx_enum pad_mode;                        /*!< \brief A VX_TYPE_ENUM of the <tt> \ref vx_pad_mode_e </tt> enumeration. */
    vx_scalar pad_const;                     /*!< \brief pad const value if setting pad mode to const, the const value is base value, not quantized value. */
} vx_nn_convolution_params_ext_t;

/*! \brief Input parameters for a deconvolution operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_deconvolution_params_t
{
    vx_size padding_x;                 /*!< \brief Number of elements subtracted at each side in the x dimension of the output. */
    vx_size padding_y;                 /*!< \brief Number of elements subtracted at each side in the y dimension of the output. */
    vx_enum overflow_policy;         /*!< \brief A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration. */
    vx_enum rounding_policy;         /*!< \brief A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration. */
    vx_size a_x;                 /*!< \brief user-specified quantity used to distinguish between the \f$upscale_x\f$ different possible output sizes. */
    vx_size a_y;                 /*!< \brief user-specified quantity used to distinguish between the \f$upscale_y\f$ different possible output sizes. */
} vx_nn_deconvolution_params_t;

/*! \brief Extended input parameter for a deconvolution operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_deconvolution_params_ext_t
{
    vx_nn_deconvolution_params_t khr;        /*!< \brief Khronos standard structure head <tt> \ref vx_nn_deconvolution_params_t <tt> */
    vx_size padding_x_right;                 /*!< \brief Number of elements subtracted at each side in the right of x dimension of the input."padding_x" is for the left */
    vx_size padding_y_bottom;                /*!< \brief  Number of elements subtracted at each side in the bottom of y dimension of the input. "padding_y" is for the top */
    vx_int32 channel_group;                  /*!< \brief  Number of separate groups for deconvolution (Range: 0 <= groups <= size of z dimension of input; size of z dimension of input can be divided by groups) */
    vx_enum pad_mode;                        /*!< \brief  A VX_TYPE_ENUM of the <tt> \ref vx_pad_mode_e <tt> enumeration. */
    vx_scalar pad_const;                     /*!< \brief  The pad const value if setting pad mode to const, the const value is base value, not quantized value. */
} vx_nn_deconvolution_params_ext_t;

typedef struct _vx_nn_deconvolution_params_ext2_t
{
    vx_nn_deconvolution_params_ext_t ext;    /*!< \brief Deconvolution extension structure head */
    vx_uint32 stride_x;                      /*!< \brief  skip x jump for down scale.  */
    vx_uint32 stride_y;                      /*!< \brief  skip y jump for down scale.  */
    vx_enum down_scale_size_rounding;        /*!< \brief Rounding method for calculating output dimensions. See <tt>\ref vx_nn_rounding_type_e</tt> */
} vx_nn_deconvolution_params_ext2_t;

/*! \brief Input parameters for ROI pooling operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_roi_pool_params_t
{
    vx_enum pool_type;  /*!< \brief Of type <tt>\ref vx_nn_pooling_type_e</tt>. Only <tt>\ref VX_NN_POOLING_MAX</tt> pooling is supported. */
} vx_nn_roi_pool_params_t;

/*! \brief Extended input parameters for ROI pooling operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_roi_pool_params_ext_t
{
    vx_nn_roi_pool_params_t     khr;                        /*!< \brief Khronos standard structure head <tt>\ref vx_nn_roi_pool_params_t</tt> */
    vx_float32                  spatial_scale;              /*!< \brief The ratio of image to feature map (Range: 0 < spatial_scale <= 1) */
    vx_int32                    pooled_height;              /*!< \brief The height of roi pooling (Range: 0 < pool_height <= height of input_data) */
    vx_int32                    pooled_width;               /*!< \brief The width of roi pooling(Range: 0 < pool_height <= width of input_data) */
} vx_nn_roi_pool_params_ext_t;

typedef struct _vx_nn_convolution_params_ext2_t
{
    vx_nn_convolution_params_ext_t ext;      /*!< \brief Convolution extension structure head */

    vx_uint32       stride_x;                /*!< \brief  skip x jump for down scale.  */
    vx_uint32       stride_y;                /*!< \brief  skip y jump for down scale.  */

    vx_int32 depth_multiplier;               /*!< \brief depthwise multiplier value, if 0, means convolution, elsewise(>=1), the convolution is depthwiseconvolution. */
} vx_nn_convolution_params_ext2_t;
/*==============================================================================
    NN Nodes
=============================================================================*/
/*! \brief [Graph] Creates a Convolutional Network Convolution Layer Node.
 * \details This function implement Convolutional Network Convolution layer.
 *  For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] = saturate(round(\sum_{l} (\sum_{m,n} inputs[j+m,k+n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Convolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for index along the width dimension and y for index along the height dimension.\n
 * before the Convolution is done, a padding with zeros of the width and height input dimensions is performed.
 * Then down scale is done by picking the results according to a skip jump. The skip in the x and y is determined by the output size dimensions.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} = round(\frac{(width_{input} + 2 * padding_x - kernel_x - (kernel_x -1) * dilation_x)}{skip_x} + 1) \f$\n
 * and \n
 * \f$ height_{output} = round(\frac{(height + 2 * padding_y - kernel_y - (kernel_y -1) * dilation_y)}{skip_y} + 1) \f$\n 
 * where \f$width\f$ is the size of the input width dimension. \f$height\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height dimensions.
 * skip is calculated by the relation between input and output. In case of ambiguity in the inverse calculation of the skip. The minimum solution is chosen. Skip must be a positive non zero integer.
 * rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>.
 * Notice that this node creation function has more parameters than the corresponding kernel. Numbering of kernel parameters (required if you create this node using the generic interface) is explicitly specified here.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches].\n 
 * \param [in] weights [*static] Weights are 4d tensor with dimensions [kernel_x, kernel_y, #IFM, #OFM]. see <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt> \n Weights data type must match the data type of the inputs.  (Kernel parameter #1)
 * \param [in] biases [*static] Optional, ignored if NULL. The biases, which may be shared (one per ofm) or unshared (one per ofm * output location). The possible layouts are
 * either [#OFM] or [width, height, #OFM]. Biases data type must match the data type of the inputs. 
 * \param [in] convolution_params [static] Pointer to parameters of type <tt>\ref vx_nn_convolution_params_t</tt>. 
 * \param [in] size_of_convolution_params [static] Size in bytes of convolution_params. Note that this parameter is not counted as one of the kernel parameters.
 * \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input. Output tensor data type must be same as the inputs.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases, const vx_nn_convolution_params_t *convolution_params, vx_size size_of_convolution_params, vx_tensor outputs);

/*! \brief [Graph] Creates a Fully connected Convolutional Network Layer Node.
* \details This function implement Fully connected Convolutional Network layers.
* In case the input and output <tt>\ref vx_tensor</tt> are signed 16. A fixed point calculation is performed with round and saturate according to the number of accumulator bits. \n
* round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
* saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
* The saturation is done based on the accumulator_bits parameter.
* According the accumulator_bits, the saturation might not be performed every operation. 
* But every a specified amount of operations, 
* that are suspected to saturate the accumulation bits\n
* The equation for Fully connected layer:\n
* \f$ outputs[i] = ( \sum_{j} saturate(round(inputs[j] \times weights[j,i])))+biasses[i] \f$\n
* Where \f$j\f$ is a index on the input feature and \f$i\f$ is a index on the output.
* before the fully connected is done, a padding of the input is performed.
* Then down scale is done by picking the results according to a skip jump. The skip is determined by the output size dimensions.
* The relation between input to output is as follows:
* \f$ size_{output} = round(\frac{(size_{input} + 2 * pad)}{skip} + 1) \f$\n
* where \f$size_{input}\f$ is the size of the input dimension. 
* \f$size_{output}\f$ is the size of the output dimension. 
* skip is calculated by the relation between input and output.
* rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>. 
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. There two possible input layouts:
* 1. [#IFM, #batches]. See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
* 2. [width, height, #IFM, #batches]. See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>\n
* In both cases number of batches are optional and may be multidimensional.
* The second option is a special case to deal with convolution layer followed by fully connected.
* The dimension order is [#IFM, #batches]. See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>. Note that batch may be multidimensional.
* \param [in] weights [*static] Number of dimensions equals dim(single input)+1. Single input dims are [width, height, #IFM], with height and #IFM being optional.\n
* \param [in] biases [*static]The biases, which may be shared (one per ofm) or unshared (one per ofm * output location).
* \param [in] pad [static] Number of elements added at each side in the input.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [out] outputs The output tensor data. Output dimension layout is [#OFM,#batches]. See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>, where #batches may be multidimensional.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxFullyConnectedLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor biases, vx_enum overflow_policy, vx_enum rounding_policy, vx_tensor outputs);

/*! \brief [Graph] Creates a Convolutional Network Local Response Normalization Layer Node. This function is optional for 8-bit extension with the extension string 'KHR_NN_8'.
 * \details Normalizing over local input regions. Each input value is divided by \f$ (\bias+\frac{\alpha}{n}\sum_i x^2_i)^\beta \f$ , where n is the number of elements to normalize across.
 * and the sum is taken over a rectangle region centred at that value (zero padding is added where necessary).
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional. Dimension layout is [width, height, IFM, #batches].
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
 * Implementations must support input tensor data types indicated by the extension strings 'KHR_NN_8 KHR_NN_16'.
 * Since this function is optional for 'KHR_NN_8', so implementations only must support <tt>VX_TYPE_INT16</tt> with fixed_point_position 8.
 * \param [in] type [static] Either same map or across maps (see <tt>\ref vx_nn_norm_type_e</tt>).
 * \param [in] normalization_size [static] Number of elements to normalize across. Must be a positive odd number with maximum size of 7 and minimum of 3.
 * \param [in] alpha [static] Alpha parameter in the local response normalization equation. must be positive.
 * \param [in] beta  [static] Beta parameter in the local response normalization equation. must be positive.
 * \param [in] bias  [static] Bias parameter in the local response normalization equation. must be positive.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
 * \ingroup group_cnn
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxLocalResponseNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_enum type,
        vx_size normalization_size,
        vx_float32 alpha,
        vx_float32 beta,
        vx_float32 bias,
        vx_tensor outputs);

/*! \brief Input parameter for normalization layer2
* \ingroup group_cnn
*\version 0.4
*/
typedef struct _vx_nn_normalization_params_t
{
    vx_enum type;   /*!< \brief Either same map or across maps <tt>\refvx_convolutional_network_norm_type_e </tt> */
    vx_uint32 norm_size; /*!< \brief Number of elements to normalize across */
    vx_float32 alpha; /*!< \brief Alpha parameter in the normalization equation */
    vx_float32 beta; /*!< \brief Beta parameter in the normalization equation */
    vx_float32 bias;  /*!< \brief Bias parameter, must not be zero */
} vx_nn_normalization_params_t;

/*! \brief extenstion parameters for normalization layer2.
 * \ingroup group_cnn
 *\version 0.5
 */
typedef struct _vx_nn_normalization_params_ext_t
{
    vx_nn_normalization_params_t base;        /*!< \brief Khronos standard structure head <tt> \ref vx_nn_normalization_params_t <tt> */
     vx_int32 axis;
} vx_nn_normalization_params_ext_t;

/*! \brief Input parameter for tensor transpose layer2
* \ingroup group_cnn
*\version 0.5
*/
typedef struct _vx_nn_transpose_params_t
{
    vx_int32* dims;      /*!< \brief The array of perm dims </tt> */
    vx_uint32 dims_num; /*!< \brief Number of dims */
} vx_nn_transpose_params_t;

/*! \brief Input parameter for tensor mean layer
* \ingroup group_cnn
*\version 0.5
*/
typedef struct _vx_nn_mean_params_t
{
    vx_tensor axis;            /*!< \brief 1D axis tensor of reduce dims </tt> */
    vx_int32 keep_dims;        /*!< \brief Keep dims, if positive, retains reduced dims with length 1 */
} vx_nn_mean_params_t;

/*! \brief Input parameter for tensor squeeze layer
* \ingroup group_cnn
*\version 0.5
*/
typedef struct _vx_nn_squeeze_params_t
{
    vx_tensor squeeze_dims;            /*!< \brief [Optional]1D tensor of squeeze dims, if specified, only squeezes the dimisions lists. otherwise, squeeze all  </tt> */
} vx_nn_squeeze_params_t;

/*! \brief Input parameter for tensor stride slice layer
* \ingroup group_cnn
*\version 0.5
*/
typedef struct _vx_nn_stride_slice_params_t
{
    vx_tensor begin_dims;        /*!< \brief 1D tensor of int32, the starts of the dims of the input tensor to be sliced. the length must be of rank(input) </tt> */
    vx_tensor end_dims;          /*!< \brief 1D tensor of int32, the ends of the dims of the input tensor to be sliced. the length must be of rank(input) </tt> */
    vx_tensor stride_dims;       /*!< \brief 1D tensor of int32, the stride of the dims of the input tensor to be sliced. the length must be of rank(input) </tt>, note that a stride can be negative, which cause a reverse slice */
    vx_int32 begin_mask;         /*!< \brief begin mask, if the ith bit of begin maks is set, begin[i] is ignored and the fullest possible range in that dim is used instead. */
    vx_int32 end_mask;           /*!< \brief end mask, if the ith bit of end maks is set, end[i] is ignored and the fullest possible range in that dim is used instead. */
    vx_int32 shrink_axis_mask;   /*!< \brief An int32 mask, if the ith bit of shrink axis mask is set, it implies that the ith specification shrinks dim must be preserved. */
} vx_nn_stride_slice_params_t;

/*! \brief [Graph] Creates a Convolutional Network Normalization Layer Node.
* \details Normalizing over local input regions. Each input value is divided by \f$ (bias+\frac{\alpha}{n}\sum_i x^2_i)^\beta \f$ , where n is the number of elements to normalize across.
:* and the sum is taken over the region centred at that value (zero padding is added where necessary).
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional.Dimension layout is [width, height, IFM, #batches].
* See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
* \param [in] nomalization_params [static] Pointer to <tt>\ref vx_nn_normalization_params_t </tt> parameter structure.
* \param [in] size_of_normalization_param [static] The size of the parameter structure.
* \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
* \ingroup group_cnn
* \version 0.4
* \return <tt> vx_node</tt>.
*/
VX_API_ENTRY vx_node VX_API_CALL vxNormalizationLayer2(vx_graph graph, vx_tensor inputs, const vx_nn_normalization_params_t *normalization_params,
    vx_size size_of_normalization_param, vx_tensor outputs);

/*! \brief [Graph] Creates a Convolutional Network Activation Layer Node.
 * The function operate a specific function (Specified in <tt>\ref vx_nn_activation_function_e</tt>), On the input data.
 * the equation for the layer is:
 * \f$ outputs(i,j,k,l) = function(inputs(i,j,k,l), a, b) \f$ for all i,j,k,l.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [in] function [static] Non-linear function (see <tt>\ref vx_convolutional_network_activation_func_e</tt>). Implementations must support <tt>\ref VX_NN_ACTIVATION_LOGISTIC</tt>, <tt>\ref VX_NN_ACTIVATION_HYPERBOLIC_TAN</tt> and <tt>\ref VX_NN_ACTIVATION_RELU</tt>
 * \param [in] a [static] Function parameters a. must be positive.
 * \param [in] b [static] Function parameters b. must be positive.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
 * \ingroup group_cnn
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxActivationLayer(vx_graph graph, vx_tensor inputs, vx_enum function, vx_float32 a,vx_float32 b, vx_tensor outputs); 

/*! \brief [Graph] Creates a Convolutional Network ROI pooling node
 * \details Pooling is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. The ROI Pooling get an array of roi rectangles, and an input tensor.
 * The kernel crop the width and height dimensions of the input tensor with the ROI rectangles and down scale the result to the size of the output tensor. The output tensor width and height are the pooled width and pooled height.
 * The down scale method is determined by the pool_type.
 * Notice that this node creation function has more parameters than the corresponding kernel. Numbering of kernel parameters (required if you create this node using the generic interface) is explicitly specified here.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional. Dimension layout is [width, height, #IFM, #batches]. 
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
 * Implementations must support input tensor data types indicated by the extension strings 'KHR_NN_8' or 'KHR_NN_8 KHR_NN_16'.  (Kernel parameter #0) 
 * \param [in] inputs_rois The roi array tensor. ROI array with dimensions [4, roi_count, #batches] where the first dimension represents 4 coordinates of the top left and bottom right corners of the roi rectangles, based on the input tensor width and height.
 * #batches is optional and must be the same as in inputs. roi_count is the number of ROI rectangles.  (Kernel parameter #1)
 * \param [in] pool_type [static] Of type <tt>\ref vx_nn_pooling_type_e</tt>. Only <tt>\ref VX_NN_POOLING_MAX</tt> pooling is supported.   (Kernel parameter #2)
 * \param [in] size_of_roi_params [static] Size in bytes of roi_pool_params. Note that this parameter is not counted as one of the kernel parameters.
 * \param [out] output_arr The output tensor. Output will have [output_width, output_height, #IFM, #batches] dimensions. #batches is optional and must be the same as in inputs.  (Kernel parameter #3)
 * \ingroup group_cnn
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxROIPoolingLayer(vx_graph graph, vx_tensor input_data, vx_tensor input_rois, const vx_nn_roi_pool_params_t *roi_pool_params, vx_size size_of_roi_params, vx_tensor output_arr);
                
                
/*! \brief [Graph] Creates a Convolutional Network Deconvolution Layer Node.
 * \details  Deconvolution denote a sort of reverse convolution, which importantly and confusingly is not actually a proper mathematical deconvolution.
 * Convolutional Network Deconvolution is up-sampling of an image by learned Deconvolution coefficients.
 * The operation is similar to convolution but can be implemented by up-sampling the inputs with zeros insertions between the inputs,
 * and convolving the Deconvolution kernels on the up-sampled result. 
 * For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] =  saturate(round(\sum_{l} \sum_{m,n}(inputs_{upscaled}[j+m,k+n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Deconvolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for the width dimension and y for the height dimension.\n
 * before the Deconvolution is done, up-scaling the width and height dimensions with zeros is performed.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} =  (width_{input} -1) * upscale_x  - 2 * padding_x + kernel_x + a_x \f$\n
 * and \n
 * \f$ height_{output} =  (height_{input} - 1) * upscale_y - 2 * padding_y + kernel_y + a_y \f$\n 
 * where \f$width_{input}\f$ is the size of the input width dimension. \f$height_{input}\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height. \f$a_x\f$ and \f$a_y\f$ are user-specified quantity used to distinguish between the \f$upscale_x\f$ and \f$upscale_y\f$ different possible output sizes.
 * \f$upscale_x\f$ and \f$upscale_y\f$ are calculated by the relation between input and output.
 * \f$a_x\f$ and \f$a_y\f$ must be positive and smaller then \f$upscale_x\f$ and \f$upscale_y\f$ respectively.
 * Since the padding parameter is on the output. The effective input padding is: \n
 * \f$ padding_{input_x} = kernel_x -padding_x -1\f$ \n
 * \f$ padding_{input_y} = kernel_y -padding_y -1\f$ \n
 * Therfore the following constarints apply : \f$kernel_x >= padding_x - 1\f$ and \f$kernel_y >= padding_y - 1\f$.
 * rounding is done according to <tt>\ref vx_nn_rounding_type_e</tt>.
 * Notice that this node creation function has more parameters than the corresponding kernel. Numbering of kernel parameters (required if you create this node using the generic interface) is explicitly specified here.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Dimension layout is [width, height, #IFM, #batches].
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
 * Implementations must support input tensor data types indicated by the extension strings 'KHR_NN_8' or 'KHR_NN_8 KHR_NN_16'.   (Kernel parameter #0)
 * \param [in] weights [static] The 4d weights with dimensions [width, height, #IFM, #OFM]. See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.  (Kernel parameter #1)
 * \param [in] biases [static] Optional, ignored if NULL. The biases have one dimension [#OFM]. Implementations must support input tensor data type same as the inputs.  (Kernel parameter #2)
 * \param [in] deconvolution_params [static] Pointer to parameters of type <tt>\ref vx_nn_deconvolution_params_t</tt>  (Kernel parameter #3)
 * \param [in] size_of_deconv_params [static] Size in bytes of deconvolution_params. Note that this parameter is not counted as one of the kernel parameters.
 * \param [out] outputs The output tensor. The output has the same number of dimensions as the input.  (Kernel parameter #4)
 * \ingroup group_cnn
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxDeconvolutionLayer(vx_graph graph, vx_tensor inputs, vx_tensor weights, vx_tensor  biases, const vx_nn_deconvolution_params_t *deconvolution_params, vx_size size_of_deconv_params, vx_tensor outputs);

/*! \brief [Graph] Creates a LeakyRELU Layer Node.
 * \details Activate the layer with leakyRELU algorithm. Given an input value x, the leakyRELU layer computes the output as x if x > 0 and negative_slope * x if x <= 0.
 * \param [in] graph The reference to the parent graph.
 * \param [in] inputs The input tensor data to reorg.
 * \param [in] negative_slope [static] specifies whether to leak the nagative part by multiplying it with the slope value rather than setting it to 0.
 * \param [in] outputs The output tensor data. Output will have same dimensions number as inputs.
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxLeakyReluLayer(
    vx_graph                    graph, 
    vx_tensor                   inputs,
    vx_float32                  negative_slope,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a PRelu Layer Node.
 * \details Activate the layer with parametric RELU algorithm. Given an input value x, the PRelu layer computes the output as x if x > 0 and alpha * x if x <= 0.
 * \param [in] graph The reference to the parent graph.
 * \param [in] inputs The input tensor data to reorg.
 * \param [in] alpha The per channel alpha tensor to leak the nagative part by multiplying it with alpha value.
 * \param [in] outputs The output tensor data. Output will have same dimensions number as inputs.
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.5
*/
VX_API_ENTRY vx_node VX_API_CALL vxPReluLayer(
    vx_graph                    graph, 
    vx_tensor                   inputs,
    vx_tensor                   alpha,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Batch Normalization Node.
 * \details Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
 * \param [in] graph The handle to the graph.
 * \param [in] eps [static] Float 32. Small value to add to the variance estimate so that we don't divide by zero.(default is 1e-5)
 * \param [in] mean [static] A mean tensor data.
 * \param [in] variance [static] A variance tensor data.
 * \param [in] gamma [static] A scale tensor data, often denoted gamma in equations.
 * \param [in] beta [static] A offset tensor data, often denoted beta in equations.
 * \param [in] input The input tensor.
 * \param [out] output The output tensor.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxBatchNormalizationLayer(
    vx_graph                    graph,
    vx_float32                  eps,
    vx_tensor                   mean,
    vx_tensor                   variance,
    vx_tensor                   gamma,
    vx_tensor                   beta,
    vx_tensor                   input,
    vx_tensor                   output
    );

/*! \brief [Graph] Creates a concat Node.
 * \details Concat one tensor from two tensor.
 * \param [in] graph The handle to the graph.
 * \param [in] in0 The input 0 tensor to be combined.
 * \param [in] in1 The input 1 tensor to be combined.
 * \param [out] out The output tensor.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConcat2Layer(
    vx_graph graph,
    vx_tensor in0,
    vx_tensor in1,
    vx_tensor out
    ); 

/*! \brief parameter for vxConcatIndefiniteLayer
 * \ingroup group_cnn
 * \version 0.4
 */
typedef struct _vx_nn_concat_params_t
{  
    vx_uint32 axis;             /*!< \brief  The axis on which we need do concat. */
} vx_nn_concat_params_t;

/*! \brief [Graph] Create a concat layer for indefinite number of tensors.
 * \param [in] graph The handle to the graph
 * \param [in] in Pointer to a list of tensors
 * \param [in] concat_params [static] Pointer to parameters of type <tt>\ref vx_nn_concat_params_t</tt>
 * \param [in] size_of_concat_params [static] Size in bytes of vx_nn_concat_params_t.
 * \param [out] out  The output tensor after concat
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConcatIndefiniteLayer(
    vx_graph graph,
    vx_object_array in,
    const vx_nn_concat_params_t* concat_params,
    vx_size size_of_concat_params,
    vx_tensor out
    );

/*! \brief The type list of reorgnization.
 * \ingroup group_cnn
 * \version 0.4
 */
enum vx_reorg_type_e
{
    /*! \brief  Reorgnization from depth to space. */
    VX_REORG_DEPTH_TO_SPACE = 0,

    /*! \brief  Reorgnization from space to depth. */
    VX_REORG_SPACE_TO_DEPTH = 1,

    /*! \brief  Reorgnization from batch to space. */
    VX_REORG_BATCH_TO_SPACE_ND,

    /*! \brief  Reorgnization from space to batch. */
    VX_REORG_SPACE_TO_BATCH_ND,

    /*! \brief Reorgnzation channel. */
    VX_REORG_SHUFFLE_CHANNEL,
};

/*! \brief Input parameter for reorg layer 
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_reorg_params_t
{
    vx_tensor block_size;           /*!< \brief  The block sizes(int32) for each spatial dimensions of the input to do a reorg operation, all value must > 1 */
    vx_enum type;                   /*!< \brief  The type of Reorgnization, <tt>\ref vx_reorg_type_e </tt> */
} vx_nn_reorg_params_t, * vx_nn_reorg_params;

/*! \brief extenstion parameters for reorg layer .
 * \ingroup group_cnn
 *\version 0.5
 */
typedef struct _vx_nn_reorg_params_ext_t
{
    vx_nn_reorg_params_t base;      /*!< \brief vx_nn_reorg_params <tt>\ref vx_nn_reorg_params_t</tt> */
    vx_tensor pad;                  /*!< \brief  [Optional] Only for SPACE2BATCH, 2D tensor for paddings for each spatial dim of the input tensor(rank(input), 2), all values must be >=0. */
} vx_nn_reorg_params_ext_t;

typedef struct _vx_nn_reorg_params_ext2_t
{
    vx_nn_reorg_params_t base;      /*!< \brief vx_nn_reorg_params <tt>\ref vx_nn_reorg_params_t</tt> */
    vx_int32 *num_group;                  
    vx_int32 *axis;
} vx_nn_reorg_params_ext2_t;

/*! \brief [Graph] Creates a Reorgnization Layer Node, Enhancement of vxReorgLayer, Support both DEPTH to SPACE and SPACE to DEPTH.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to reorg.
 * \param [in] reorg_params [static] Pointer to parameters of type <tt>\ref vx_nn_reorg_params</tt>
 * \param [in] size_of_reorg_params [static] Size in bytes of vx_nn_reorg_params.
 * \param [out] output The output tensor data. Output will have different number of each dimensions as input.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxReorgLayer2(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_reorg_params    reorg_params,
    vx_size                     size_of_reorg_params,
    vx_tensor                   output
    );

/*! \brief Input parameter for TensorRoundingLayer
 * \ingroup group_tensor
 * \version 0.4
 */
typedef struct _vx_nn_rounding_params_t
{
    vx_enum mode;           /*!< \brief  Rounding method for calculating tensor data(VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_FLOOR or VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING). See <tt>\ref vx_convolutional_network_rounding_type_e</tt> */
} vx_nn_rounding_params_t, * vx_nn_rounding_params;

/*! \brief [Graph] Creates a Rounding Layer Node, support FLOOR and CEIL.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to reorg.
 * \param [in] rounding_params [static] Pointer to parameters of type <tt>\ref vx_nn_rounding_params</tt>
 * \param [in] size_of_rounding_params [static] Size in bytes of vx_nn_rounding_params.
 * \param [out] output The output tensor data.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_tensor
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorRoundingNode(
    vx_graph                       graph, 
    vx_tensor                      input,
    const vx_nn_rounding_params    rounding_params,
    vx_size                        size_of_rounding_params,
    vx_tensor                      output
    );

/*! \brief Input parameter for hashTableLookupLayer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_hashlut_params_t
{
    vx_tensor keys;             /*!< \brief  A 1-D tensor with shape [ n ]; */
    vx_tensor values;           /*!< \brief  A tensor with shape of [ n, ?]; i.e., the first dimension must be n. */
} vx_nn_hashlut_params_t, * vx_nn_hashlut_params;

/*! \brief [Graph] Creates a hash lookup table Layer Node.
 * \details  Keys and Values pair represent a map, i.e., the ith element
 *           in Keys (Keys[i]) is the key to select the ith sub-tensor
 *           in Values (Values[i]), where 0 <= i <= n-1.
 *           Keys tensor *MUST* be sorted in ascending order.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input 1-D tensor with shape [ k ].
 * \param [in] hashlut_params Pointer to parameters of type <tt>\ref vx_nn_hashlut_params_t</tt>
 * \param [in] size_of_hashlut_params [static] Size in bytes of vx_nn_hashlut_params.
 * \param [out] hits A boolean tensor with shape [ k ] indicates whether the lookup hits (True) or not (False).
 * \param [out] output The output tensor data, tensor with shape [ k, ?]
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxHashTableLookupLayer(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_hashlut_params  hashlut_params,
    vx_size                     size_of_hashlut_params,
    vx_tensor                   hits,
    vx_tensor                   output
    );

/*! \brief  LSH project type list
 *\ingroup group_cnn
 *\version 0.4
 */
enum vx_lshproj_type_e {
    /*! \brief  Computed bit vector is considered to be sparse. */
    VX_LSH_PROJ_SPARSE = 1,

    /*! \brief  Computed bit vector is considered to be dense. */
    VX_LSH_PROJ_DENSE = 2,
};

/*! \brief Input parameter to LSH projection layer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_lshproj_params_t
{
    vx_tensor hash_func;         /*!< \brief  Tensor of hash function. Dim size is 2, .Dim[0]: Number of hash functions. Dim[1]: Number of seeds per hash functions. Dim[1] <= 32 in sparse case. */
    vx_tensor weights;           /*!< \brief  Optional. Dim.size == 1, If not set, each input element is considered to have the same weight of 1.0. */
    vx_tensor type;                /*!< \brief  The type of LSH projection, support VX_LSH_PROJ_SPARSE and VX_LSH_PROJ_DENSE; */
} vx_nn_lshproj_params_t, * vx_nn_lshproj_params;

/*! \brief [Graph] Creates a LSH projection Layer Node.
 * \details  Projects an input to a bit vector via locality senstive hashing.
 *   Sparse: Value VX_LSH_PROJ_SPARSE(=1).
 *     Computed bit vector is considered to be sparse.
 *     Each output element is an int32 made up of multiple bits computed from
 *     hash functions.
 *   Dense: Value VX_LSH_PROJ_DENSE(=2).
 *     Computed bit vector is considered to be dense. Each output element
 *     represents a bit and can take the value of either 0 or 1.
 *
 * \param [in] graph The reference to the parent graph.
 * \param [in] input input tensor data, Dim size must >= 1.
 * \param [in] lshproj_params Pointer to parameters of type <tt>\ref vx_nn_lshproj_params</tt>
 * \param [in] size_of_lshproj_params [static] Size in bytes of vx_nn_lshproj_params.
 * \param [out] output The output tensor data.
 *  If the projection type is sparse: 
 *    Output.Dim == { Tensor[0].Dim[0] }
 *    A tensor that represents hash signatures.
 *  If the projection type is Dense:
 *    Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
 *    A flattened tensor that represents projected bit vectors.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxLSHProjectionLayer(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_lshproj_params  lshproj_params,
    vx_size                     size_of_lshproj_params,
    vx_tensor                   output
    );

/*! \brief Input parameter for Reshape layer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_reshape_params_t
{
    vx_tensor dims;                    /*!< \brief  dimension. */  
} vx_nn_reshape_params_t, * vx_nn_reshape_params;

/*! \brief [Graph] Creates a Reshape Layer Node.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to reshape.
 * \param [in] reshape_params Pointer to parameters of type <tt>\ref vx_nn_reshape_params</tt>
 * \param [in] size_of_reshape_params [static] Size in bytes of vx_nn_reshape_params.
 * \param [out] output The output tensor data.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_tensor
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorReshapeNode(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_reshape_params  reshape_params,
    vx_size                     size_of_reshape_params,
    vx_tensor                   output
    );

/*! \brief Input parameter for Scale layer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_scale_params_t
{
    vx_enum type;             /*!< \brief  The interpolation type, only support VX_INTERPOLATION_BILINEAR.  */
} vx_nn_scale_params_t, * vx_nn_scale_params;

/*! \brief [Graph] Creates a scale Layer Node.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to scale.
 * \param [in] scale_params [static] Pointer to parameters of type <tt>\ref vx_nn_scale_params</tt>
 * \param [in] size_of_scale_params [static] Size in bytes of vx_nn_scale_params.
 * \param [out] output The output tensor data.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_tensor
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorScaleNode(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_scale_params    scale_params,
    vx_size                     size_of_scale_params,
    vx_tensor                   output
    );

/*! \brief Input parameter for YUV to RGB scale layer
 *\ingroup group_cnn
 *\version 0.5
 */
typedef struct _vx_nn_yuv2rgb_scale_params_t
{
    vx_rectangle_t rect;    /*!< \brief  The rectangle region of input image to do yuv2rgb scale. If it is set to 0, region is full input image; */
    vx_float32  mean_r;     /*!< \brief  Mean coefficient for output r channel; */
    vx_float32  mean_g;     /*!< \brief  Mean coefficient for output g channel; */
    vx_float32  mean_b;     /*!< \brief  Mean coefficient for output b channel; */
    vx_float32  scale_rgb;  /*!< \brief  Scale coefficient value for output rgb; Not the scale ratio; */
    vx_bool     y_only;     /*!< \brief  YUV mode, Y only or normal YUV.  */
    vx_bool     output_rgb; /*!< \brief  Output mode, BGR or RGB.  */
    vx_bool     output_roi; /*!< \brief  Output full image or partial region of image. Default is full image. */
    vx_uint8    fill_r;     /*!< \brief  R channel value of output image pad. */
    vx_uint8    fill_g;     /*!< \brief  G channel value of output image pad. */
    vx_uint8    fill_b;     /*!< \brief  B channel value of output image pad. */
    vx_rectangle_t output_rect; /*!< \brief  The rectangle region of output image. It should be smaller than input image. If output_roi is false, this parameter will be ignored.*/
} vx_nn_yuv2rgb_scale_params_t, * vx_nn_yuv2rgb_scale_params;

/*! \brief [Graph] Creates a scale Layer Node.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to scale.
 * \param [in] scale_params [static] Pointer to parameters of type <tt>\ref vx_nn_scale_params</tt>
 * \param [in] size_of_scale_params [static] Size in bytes of vx_nn_scale_params.
 * \param [out] output The output tensor data.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_tensor
 * \version 0.5
 */
VX_API_ENTRY vx_node VX_API_CALL vxYUV2RGBScaleNode(
    vx_graph                          graph,
    vx_image                          input,
    const vx_nn_yuv2rgb_scale_params  yuv2rgb_scale_params,
    vx_size                           size_of_yuv2rgb_scale_param,
    vx_tensor                         output
    );

/*! \brief Input parameter for RNN layer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_rnn_params_t
{
    vx_tensor weights;              /*!< \brief 2-D recurrent weights tensor, of shape [num_units, input_size], where "num_units" corresponds to the number of units.  */
    vx_tensor recurrent_weights;    /*!< \brief 2-D tensor, of shape [num_units, num_units], with columns corresponding to the weights from each unit.  */
    vx_tensor bias;                 /*!< \brief 1-D tensor, of shape [num_units].  */
    vx_tensor state_in;             /*!< \brief 2-D tensor, of shape [batch_size, num_units].  */
    vx_tensor activation;           /*!< \brief Optional, indicating the activation function. If "NONE" is specified then it results in a linear activation. */
} vx_nn_rnn_params_t, * vx_nn_rnn_params;

/*! \brief [Graph] Creates a RNN Layer Node.
 * \details A basic recurrent neural network layer.
 *      This layer implements the operation:
 *      outputs = state = activation(inputs * input_weights + state * recurrent_weights + bias)
 *      
 *      Where:
 *      "input_weights" is a weight matrix that multiplies the inputs;
 *      "recurrent_weights" is a weight matrix that multiplies the current
 *       "state" which itself is the output from the previous time step
 *       computation;
 *      "bias" is a bias vector (added to each output vector in the batch);
 *      "activation" is the function passed as the "activation_function"
 *      argument (if not "NONE").
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data to rnn, 2-D tensor, of shape [input_size, batch_size], where "batch_size" corresponds to the batching dimension, and "input_size" is the size of the input.
 * \param [in] rnn_params Pointer to parameters of type <tt>\ref vx_nn_rnn_params</tt>
 * \param [in] size_of_rnn_params [static] Size in bytes of vx_nn_rnn_params.
 * \param [out] state_out The output tensor data, A 2-D tensor, of shape [batch_size, num_units].
 * \param [out] output The output tensor data, 2-D tensor, of shape [batch_size, num_units]. This is effectively the same as the current state value..
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxRNNLayer(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_rnn_params      rnn_params,
    vx_size                     size_of_rnn_params,
    vx_tensor                   state_out,
    vx_tensor                   output
    );

/*! \brief Input parameter for softmax layer2
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_softmax_params_t
{
    vx_float32 beta;             /*!< \brief  A FLOAT32 value, specifying the positive scaling factor for the exponent, beta.  */
} vx_nn_softmax_params_t, * vx_nn_softmax_params;

/*! \brief extenstion parameters for softmax layer2.
 * \ingroup group_cnn
 *\version 0.5
 */
typedef struct _vx_nn_softmax_params_ext_t
{
    vx_nn_softmax_params_t base;        /*!< \brief Khronos standard structure head <tt> \ref vx_nn_softmax_params_t <tt> */
    vx_int32 axis;
} vx_nn_softmax_params_ext_t;

/*! \brief [Graph] Creates a softmax Layer Node.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data, with number of dimensions equals dim(input batch) + 1. Softmax will be calculated per IFM..
 * \param [in] softmax_params [static] Pointer to parameters of type <tt>\ref vx_nn_softmax_params</tt>
 * \param [in] size_of_softmax_params [static] Size in bytes of vx_nn_softmax_params.
 * \param [out] output The output tensor data, Outputs will have the same number of dimensions as input..
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxSoftmaxLayer2(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_softmax_params  softmax_params,
    vx_size                     size_of_softmax_params,
    vx_tensor                   output
    );

/*! \brief Input parameter for SVDF layer
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_svdf_params_t
{
    vx_tensor weights_feature;      /*!< \brief A 2-D tensor, of shape [num_units, input_size], where "num_units" corresponds to the number of units.  */
    vx_tensor recurrent_time;       /*!< \brief A 2-D tensor, of shape [num_units, memory_size], where "memory_size" corresponds to the fixed-size of the memory.  */
    vx_tensor bias;                 /*!< \brief Optional, 1-D tensor of type T, of shape [num_units].  */
    vx_tensor state_in;             /*!< \brief  A 2-D tensor, of shape [(memory_size - 1) * num_units * rank, batch_size]  */
    vx_tensor rank;                 /*!< \brief The rank of the SVD approximation.  */
    vx_tensor activation;           /*!< \brief Indicating the activation function, specify linear activation for default */
} vx_nn_svdf_params_t, * vx_nn_svdf_params;

/*! \brief [Graph] Creates a svdf Layer Node.
 * \details SVDF op is a kind of stateful layer derived from the notion that a
 *          densely connected layer that's processing a sequence of input frames can
 *          be approximated by using a singular value decomposition of each of its
 *          nodes. The implementation is based on:
 *        
 *          https://research.google.com/pubs/archive/43813.pdf
 *          
 *          P. Nakkiran, R. Alvarez, R. Prabhavalkar, C. Parada.
 *          "Compressing Deep Neural Networks using a Rank-Constrained Topology".
 *          INTERSPEECH, 2015.
 *          
 *          It processes the incoming input using a 2-stage filtering mechanism:
 *          stage 1 performs filtering on the "features" dimension, whose outputs get
 *          pushed into a memory of fixed-size memory_size.
 *          stage 2 performs filtering on the "time" dimension of the memory_size
 *          memoized outputs of stage 1.
 *          
 *          Specifically, for rank 1, this layer implements the operation:
 *          
 *             memory = push(conv1d(inputs, weights_feature, feature_dim,
 *                           "PADDING_VALID"));
 *             outputs = activation(memory * weights_time + bias);
 *          
 *          Where:
 *          "weights_feature" is a weights matrix that processes the inputs (by
 *          convolving the input with every "feature filter"), and whose outputs get
 *          pushed, stacked in order, into the fixed-size "memory" (the oldest entry
 *          gets dropped);
 *          "weights_time" is a weights matrix that processes the "memory" (by a
 *          batched matrix multiplication on the num_units);
 *          "bias" is an optional bias vector (added to each output vector in the
 *          batch); and
 *          "activation" is the function passed as the "fused_activation_function"
 *          argument (if not "NONE").
 *          
 *          Each rank adds a dimension to the weights matrices by means of stacking
 *          the filters.
 * \param [in] graph The reference to the parent graph.
 * \param [in] input The input tensor data, A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "batch_size" corresponds to the batching dimension, and "input_size" is
 *                    the size of the input.
 * \param [in] svdf_params Pointer to parameters of type <tt>\ref vx_nn_svdf_params</tt>
 * \param [in] size_of_svdf_params [static] Size in bytes of vx_nn_svdf_params.
 * \param [out] state_out A 2-D tensor, of shape [(memory_size - 1) * num_units * rank, batch_size].
 * \param [out] output The output tensor data, Outputs will have the same number of dimensions as input.
 * \returns <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 * \version 0.4
 */
VX_API_ENTRY vx_node VX_API_CALL vxSVDFLayer(
    vx_graph                    graph, 
    vx_tensor                   input,
    const vx_nn_svdf_params     svdf_params,
    vx_size                     size_of_svdf_params,
    vx_tensor                   state_out,
    vx_tensor                   output
    );

/*! \brief Input parameter for Pooling layer2
 *  \ingroup group_cnn
 */
typedef struct _vx_nn_pooling_params_t
{
    vx_enum pool_type;              /*!< \brief either max pooling or average pooling, see <tt>\ref vx_convolutional_network_pooling_type_e</tt>. */
    vx_uint32 pool_size_x;          /*!< \brief Size of the pooling region in the x dimension. */
    vx_uint32 pool_size_y;          /*!< \brief Size of the pooling region in the y dimension. */
    vx_uint32 pool_pad_x_left;      /*!< \brief Padding size in the left of x dimension. */
    vx_uint32 pool_pad_x_right;     /*!< \brief Padding size in the right of x dimension. */
    vx_uint32 pool_pad_y_top;       /*!< \brief Padding size in the top of y dimension. */
    vx_uint32 pool_pad_y_bottom;    /*!< \brief Padding size in the bottom of y dimension. */
    vx_enum rounding;               /*!< \brief Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt> */
} vx_nn_pooling_params_t;


/*! \brief Extended input parameter for Pooling layer2
 * \ingroup group_cnn
 * \version 0.4
 */
typedef struct _vx_nn_pooling_params_ext_t
{  
    vx_nn_pooling_params_t base;    /*!< \brief The base definition.<tt>\ref vx_nn_pooling_params_t</tt> */
    vx_uint32 stride_x;             /*!< \brief  Skip x jump for down scale. */
    vx_uint32 stride_y;             /*!< \brief  Skip y jump for down scale. */
} vx_nn_pooling_params_ext_t;


/*! \brief [Graph] Creates a Convolutional Network Pooling Layer Node, this function can support uneven padding.
 * \details Pooling is done on the first 2 dimensions or the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for the first dimension and y for the second.\n
 * Pooling operation is a function operation over a rectangle size and then a nearest neighbour down scale.
 * Here we use pool_size_x and pool_size_y to specify the rectangle size on which the operation
 * is performed. \n
 * before the operation is done (average or maximum value). the data is padded in the first 2D with zeros.
 * The down scale is done by picking the results according to a skip jump. The skip in the x and y dimension is determined by the output size dimensions.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, 4th dimension for batch of inputs is optional.Dimension layout is [width, height, #IFM, #batches].
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>
 * \param [in] pooling_params [static] Pointer to parameters of type <tt>\ref vx_nn_pooling_params_t</tt>
 * \param [in] size_of_pooling_params [static] Size in bytes of pooling_params.
 * \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxPoolingLayer2(
    vx_graph                    graph,
    vx_tensor                   inputs,
    const vx_nn_pooling_params_t * pooling_params,
    vx_size                     size_of_pooling_params,
    vx_tensor                   outputs);

/*! \brief [Graph] Performs arithmetic addition on element values in the input tensor data's.
 * \param [in] graph The handle to the graph.
 * \param [in] in1 input tensor data,. 
 * \param [in] in2 input tensor data, inputs must be of equal in dimensions.
 * else, If in one of the vx_mddata dimension is 1.
 * That dimension is considered as a const on all the dimension terms.
 * And will perform as if the values are duplicated on all terms in that dimensions.
 * After the expansion. The dimensions are equal.
 * \param [in] scale [static] The scale value.
 * \param [in] overflow_policy [static] A vx_convert_policy_e enumeration.
 * \param [in] rounding_policy [static] A vx_round_policy_e enumeration.
 * \param [out] out The output tensor data with the same dimensions as the input tensor data's.
 * \ingroup group_tensor
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorDivideNode(vx_graph graph, vx_tensor in1, vx_tensor in2, vx_scalar scale, vx_enum overflow_policy, vx_enum rounding_policy, vx_tensor out);

/*! \brief [Graph] Performs LUT on element values in the input tensor data's.
 * \param [in] graph The handle to the graph.
 * \param [in] in1 input tensor data.
 * \param [in] lut lut tensor data.
 * \param [out] out The output tensor data with the same dimensions as the input tensor data's.
 * \ingroup group_tensor
 * \return <tt> vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorTableLookupNode2(vx_graph graph, vx_tensor in1, vx_tensor lut, vx_tensor out);

/*! \brief [Graph] Performs matrices transformation on input tensor.
* The node transpose the tensor according to the matrices that perm gives.
* \param [in] graph The handle to the graph.
* \param [in] in input tensor data,
* \param [out] out output tensor data,
* \param [in] perm [static] that is the matrices to transpose. If not given, do full reversed transpose according to the input tensor dimension.
* \param [in] sizes_of_perm [static] that is the dimension of perm.
* \ingroup group_tensor
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorPermuteNode(vx_graph graph, vx_tensor in, vx_tensor out, vx_uint32* perm, vx_uint32 sizes_of_perm);

/*! \brief [Graph] Computes the sum of elements across dimensions of input tensor.
* \param [in] graph The handle to the graph.
* \param [in] in input tensor data,
* \param [out] out output tensor data,
* \param [in] reduce_dim [static] used to determine sum across which dimension(dimension 0 means width, etc). If not given, compute the sum across all dimensions.
* \param [in] dim_size [static] used to specify the array size of redume_dim.
* \param [in] keep_dim [static] means if keep the dimesion count.
* \ingroup group_tensor
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \version 0.3
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorReduceSumNode(vx_graph graph, vx_tensor in, vx_tensor out, vx_uint32* reduce_dim, vx_int32 dim_size, vx_bool keep_dim);


/*! \brief Input parameter structure for TensorPadNode
 * \ingroup group_tensor
 * \version 0.3
 */
typedef struct _vx_nn_pad_params_t
{
    vx_int32 * pad_front_array;           /*!< \brief An array of values which specify how many values are added on the front(left, top etc) of a tensor. */
    vx_int32 * pad_back_array;            /*!< \brief An array of values which specify how many values are added on the back(right, bottom etc) of a tensor. */
    vx_uint8   numViewDimensions;         /*!< \brief The size of two arrays. */
    vx_enum    pad_mode;                  /*!< \brief A VX_TYPE_ENUM of the <tt>\ref vx_pad_mode_e</tt> enumeration. */
    vx_scalar  pad_const;                 /*!< \brief The order const value if setting pad mode to const, the const value is base value, not quantized value. */

} vx_nn_pad_params_t, * vx_nn_pad_params;


/*! \brief [Graph] Performs padding on input tensor with diffrent pad mode.
* \param [in] graph The handle to the graph.
* \param [in] in input tensor data,
* \param [out] out output tensor data,
* \param [in] pad_params [static] contains pad left, right, top, bottom, pad mode, const value, etc.
* \param [in] size_of_pad_params [static] The size of pad_params.
* \ingroup group_tensor
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \version 0.3
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorPadNode(vx_graph graph, vx_tensor in, vx_tensor out, const vx_nn_pad_params pad_params, vx_size size_of_pad_params);

/*! \brief [Graph] Performs copy from source tensor to destination tensor.
*\details This copy function also perform format converion if src tensor and dst tensor have differnt formats.
* Dequatization could be done by this function.
* \param [in] graph The handle to the graph.
* \param [in] src input tensor data,
* \param [out] dst output tensor data.
* \note that copy size is the min(srcSize, dstSize)
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_tensor
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorCopyNode(vx_graph graph, vx_tensor src, vx_tensor dst);

/*! \brief Input parameter for vxTensorReverse
 * \ingroup group_cnn
 */
typedef struct _vx_nn_tensor_reverse_params_t
{
    vx_int32 *axis;  /*!< \brief array of axis */
    vx_uint32 numberOfAxis; /*!< \brief size of axis, max value is 4 */
}
vx_nn_tensor_reverse_params_t;

/*! \brief [Graph] Performs reverse on input tensor.
* \param [in] graph The handle to the graph.
* \param [in] inputs input tensor data.
* \param [in] tensor_reverse_params [static] Pointer to parameters of type <tt>\ref vx_nn_tensor_reverse_params_t</tt>.
* \param [in] size_of_tensor_reverse_params [static] The size of tensor_reverse_params.
* \param [out] outputs output tensor data.
* \ingroup group_tensor
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorReverse(vx_graph graph, vx_tensor inputs, const vx_nn_tensor_reverse_params_t * tensor_reverse_params, vx_size size_of_tensor_reverse_params, vx_tensor outputs);

/*! \brief Input parameter for L2Normalize layer2
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_l2norm_params_t
{
    vx_int32       axis;
} vx_nn_l2norm_params_t;

/*! \brief [Graph] Creates a Convolutional Network L2Normalize Layer2 Node.
 * \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor. 3 lower dimensions represent a single input, and an optional 4th dimension for batch of inputs. Dimension layout is [width, height, #IFM, #batches].
 * See <tt>\ref vxCreateTensor2</tt> and <tt>\ref vxCreateVirtualTensor2</tt>.
* \param [in] l2norm_params [static] Pointer to parameters of type <tt>\ref vx_nn_l2norm_params</tt>
* \param [in] size_of_l2norm_params [static] Size in bytes of vx_nn_l2norm_params.
* \param [out] outputs The output tensor data. Output will have the same number of dimensions as input.
* \ingroup group_cnn
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
*/
VX_API_ENTRY vx_node VX_API_CALL vxL2NormalizeLayer2(
    vx_graph                      graph, 
    vx_tensor                     inputs, 
    const vx_nn_l2norm_params_t * l2norm_params, 
    vx_size                       size_of_l2norm_params,
    vx_tensor                     outputs);

/*! \brief Input parameter structure for RPNLayer
 *\ingroup group_cnn
 */
typedef struct _vx_nn_rpn_params_t
{
    vx_uint32       feature_stride;  /*!< \brief Image feature stride.  */
    vx_uint32       min_size;        /*!< \brief The smallest rectangular box size */
    vx_uint32       pre_nms_topn;    /*!< \brief Before NMS, take pre_nms_topn rectangulars for NMS. */
    vx_uint32       post_nms_topn;   /*!< \brief After NMS, take post_nms_topn rectangulars for proposals output */
    vx_float32      nms_thresh;      /*!< \brief The IOU threshold */
} vx_nn_rpn_params_t;

/*! \brief [Graph] Creates a Regin Proposal Networks Layer Node.
 * \details A Region Proposal Network(RPN) takes an image(of any size) as input and outputs a set of rectangular object proposals,
 * each with an objectness socre.
 * \param [in] graph The handle to the graph.
 * \param [in] score The score tensor data. its has 2 types of values: foreground and background. Only foreground objects are needed.
 * \param [in] bbox The bounding box regressor tensor data. Used for bounding box regression.
 * \param [in] anchors The anchor box tensor data. A set of rectangles generated by scale and aspect ratio.
 * \param [in] img_info [static] The image information tensor data. 4 elements: image width, image height, image width scale, image height scale.
 * \param [in] rpn_params [static] Pointer to parameters of type <tt>\ref vx_nn_rpn_params_t</tt>
 * \param [in] size_of_rpn_params [static] Size in bytes of vx_nn_rpn_params.
 * \param [in] roi_output The output tensor. The proposals output tensor data. This information used by ROI pooling
 * \param [in] score_output The output tensor. The proposals score output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxRPNLayer(
    vx_graph                    graph, 
    vx_tensor                   score,
    vx_tensor                   bbox,
    vx_tensor                   anchors,
    vx_tensor                   img_info,
    const vx_nn_rpn_params_t *  rpn_params,
    vx_size                     size_of_rpn_params,
    vx_tensor                   roi_output,
    vx_tensor                   score_output
    );

/*! \brief Input parameters for a lstm operation.
 * \ingroup group_cnn
 * \version 0.3
 */
typedef struct _vx_nn_lstm_params_t
{
    vx_tensor input2input_weight;                  /*!< \brief Optional A 2-D tensor of type T, of shape [num_units, input_size]. where "num_units" corresponds to the number of cell units.*/
    vx_tensor input2forget_weight;                 /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    vx_tensor input2cell_weight;                   /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    vx_tensor input2output_weight;                 /*!< \brief  A 2-D tensor of type T, of shape [num_units, input_size].*/
    
    vx_tensor recurrent2input_weight;              /*!< \brief Optional A 2-D tensor of type T, of shape [num_units, output_size]. where "output_size" corresponds to either the number of cell units (i.e., "num_units"), or the second dimension of the "projection_weights", if defined.*/
    vx_tensor recurrent2forget_weight;             /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    vx_tensor recurrent2cell_weight;               /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    vx_tensor recurrent2output_weight;             /*!< \brief  A 2-D tensor of type T, of shape [num_units, output_size].*/
    
    vx_tensor cell2input_weight;                   /*!< \brief Optional A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor cell2forget_weight;                  /*!< \brief Optional A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor cell2output_weight;                  /*!< \brief Optional A 1-D tensor of type T, of shape [num_units].*/
    
    vx_tensor input_gate_bias;                     /*!< \brief Optional A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor forget_gate_bias;                    /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor cell_bias;                           /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
    vx_tensor output_gate_bias;                    /*!< \brief  A 1-D tensor of type T, of shape [num_units].*/
    
    vx_tensor projection_weight;                   /*!< \brief Optional A 2-D tensor of type T, of shape [output_size, num_units].*/
    vx_tensor projection_bias;                     /*!< \brief Optional A 1-D tensor of type T, of shape [output_size].*/
    
    vx_tensor activation;                          /*!< \brief Optional. An ActivationFunctionType indicating the activation function. If "NONE" is specified then it results in a linear activation.If "NONE" is specified then it results in a linear activation.*/
    vx_tensor cell_clip;                           /*!< \brief  A clipping threshold for the cell state, such that values are bound within [-cell_clip, cell_clip]. If set to 0.0 then clipping is disabled.*/
    vx_tensor proj_clip;                           /*!< \brief  A clipping threshold for the output from the projection layer, such that values are bound within [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.*/
} vx_nn_lstm_params_t;

/*! \brief extenstion parameters for a lstm unit operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_lstm_params_ext_t
{
    vx_nn_lstm_params_t base;              /*!< \brief standard structure head</tt>.*/
    vx_tensor forget_bias;                 /*!< \brief  A bias(float 32) for the forget gate. If set to 0.0f(by default) then bias is ignored.*/

    vx_float32 norm_gain;                  /*!< \brief  Float32[static] The layer normalization gain initial value(default is 1.0f).*/
    vx_float32 norm_shift;                 /*!< \brief  Float32[static] The layer normalization shift initial value(default is 0.0f).*/ 

    vx_tensor sequence_length;             /*!< \brief  Optional[static] Specifies the length of each sequence in inputs. An `int32` (tensor) size `[batch_size]`, values in `[0, time_len)` or None(by default).*/ 

    /*Since ANDROID NN API level 29 there are additional inputs to this op:*/
    vx_tensor layernorm2input_weight;              /*!< \brief [Optional] The input layer normalization weights. A 1 - D tensor of shape[num_units].Used to rescale normalized inputs to activation at input gate.*/
    vx_tensor layernorm2forget_weight;             /*!< \brief [Optional] The forget layer normalization weights. A 1 - D tensor of shape[num_units].Used to rescale normalized inputs to activation at forget gate.*/
    vx_tensor layernorm2cell_weight;               /*!< \brief [Optional] The cell layer normalization weights. A 1 - D tensor of shape[num_units].Used to rescale normalized inputs to activation at cell gate.*/
    vx_tensor layernorm2output_weight;             /*!< \brief [Optional] The output layer normalization weights. A 1 - D tensor of shape[num_units].Used to rescale normalized inputs to activation at output gate.*/
} vx_nn_lstm_params_ext_t;

/*! \brief input parameters for a lstm layer operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_lstm_layer_params_t
{
    vx_nn_lstm_params_t lstm_param;              /*!< \brief  lstm input param <tt>\ref vx_nn_lstm_params_t</tt>.*/
    vx_enum             lstm_layer_type;         /*!< \brief  lstm layer type.*/
} vx_nn_lstm_layer_params_t;

/*! \brief input parameters for a lstm layer operation.
 * \ingroup group_cnn
 */
typedef struct _vx_nn_lstm_layer_params_ext_t
{
    vx_nn_lstm_params_ext_t lstm_param;          /*!< \brief  lstm input param <tt>\ref vx_nn_lstm_params_ext_t</tt>.*/
    vx_enum             lstm_layer_type;         /*!< \brief  lstm layer type.*/
} vx_nn_lstm_layer_params_ext_t;

/*! \brief [Graph] Creates a Long short-term memory unit (LSTM) Unit Networks Layer Node.
 * \details
 *     The default non-peephole implementation is based on:
 *     http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
 *     S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
 *     Computation, 9(8):1735-1780, 1997.
 *
 *     The peephole implementation is based on:
 *     https://research.google.com/pubs/archive/43905.pdf
 *     Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
 *     recurrent neural network architectures for large scale acoustic modeling."
 *     INTERSPEECH, 2014.
 *     
 *     The coupling of input and forget gate (CIFG) is based on:
 *     http://arxiv.org/pdf/1503.04069.pdf
 *     Greff et al. "LSTM: A Search Space Odyssey"
 *     
 *     The class has the following independently optional inputs:
 *     * If input gate (if CIFG): "input_to_forget_weights",
 *       "recurrent_to_input_weights", "cell_to_input_weights", "input_gate_bias".
 *     * If no peephole connections: "cell_to_input_weights",
 *       "cell_to_forget_weights", "cell_to_output_weights".
 *     * If no projection layer: "projection_weights" and "projection_bias".
 *     * If no projection bias: "projection_bias".
 *
 * \param [in] graph The handle to the graph.
 * \param [in] input A 2-D tensor of type T, of shape [input_size, batch_size], where
 *                    "batch_size" corresponds to the batching dimension, and "input_size"
 *                    is the size of the input.
 * \param [in] output_state_in A 2-D tensor of type T, of shape [output_size, batch_size].
 * \param [in] cell_state_in A 2-D tensor of type T, of shape [num_units, batch_size].
 * \param [in] lstm_params LSTM paraments <tt>\ref vx_nn_lstm_params_t </tt>.
 * \param [in] size_of_lstm_params [static] The size of the lstm_params.
 * \param [out] scratch A 3-D tensor of type T, of shape [num_cell, 4, batch_size].
 * \param [out] output_state_out A 2-D tensor of type T, of shape [output_size, batch_size].
 * \param [out] cell_state_out A 2-D tensor of type T, of shape [num_units, batch_size].
 * \param [out] output A 2-D tensor of type T, of shape [output_size, batch_size]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.3
 */
VX_API_ENTRY vx_node VX_API_CALL vxLstmUnitLayer(
    vx_graph graph,
    vx_tensor input,
    vx_tensor output_state_in,
    vx_tensor cell_state_in,
    const vx_nn_lstm_params_t * lstm_params,
    vx_size size_of_lstm_params,
    vx_tensor scratch,
    vx_tensor output_state_out,
    vx_tensor cell_state_out,
    vx_tensor output);

/*! \brief [Graph] Creates a Long short-term memory layer (LSTM) Networks Layer Node.
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
 * \param [in] lstm_layer_params LSTM paraments <tt>\ref vx_nn_lstm_layer_params_t </tt>.
 * \param [in] size_of_lstm_layer_params [static] The size of the lstm_layer_params.
 * \param [out] output A 2-D/3D tensor of type T, of shape [output_size, batch_size] or [output_size, batch_size, time]. 
 *                      This is effectively the same as the current "output_state" value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 * \version 0.3
 */
VX_API_ENTRY vx_node VX_API_CALL vxLstmLayer(
    vx_graph graph, 
    vx_tensor input,
    vx_tensor static_input,
    vx_tensor cont,
    const vx_nn_lstm_layer_params_t * lstm_layer_params,
    vx_size size_of_lstm_layer_params,
    vx_tensor output
    );

/*! \brief [Graph] Creates transpose layer node.
* \details
*    Transposes the input tensor, permuting the dimensions according to perm tensor.
*
* \param [in] graph The handle to the graph.
* \param [in] input A n-D tensor, specifying the tensor to be transposed.
* \param [in] transpose_params paraments <tt>\ref vx_nn_transpose_params_t </tt>.
* \param [in] size_of_transpose_param [static] The size of the vx_nn_transpose_params_t.
* \param [out] output A n-D tensor of the same type as input.
* \return <tt> vx_node</tt>.
* \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_tensor
* \version 0.5
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorTransposeNode2(
    vx_graph graph,
    vx_tensor inputs,
    const vx_nn_transpose_params_t *transpose_params,
    vx_size size_of_transpose_param,
    vx_tensor outputs);

/*! \brief [Graph] Creates mean layer node.
* \details
*    Computes the mean of elements across dimensions of a tensor.
*
* \param [in] graph The handle to the graph.
* \param [in] input A n-D tensor, specifying the input.
* \param [in] mean_params paraments <tt>\ref vx_nn_mean_params_t </tt>.
* \param [in] size_of_mean_param [static] The size of the vx_nn_mean_params_t.
* \param [out] output A n-D tensor of the same type as input.
* \return <tt> vx_node</tt>.
* \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_tensor
* \version 0.5
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorMeanNode(
    vx_graph graph,
    vx_tensor inputs,
    const vx_nn_mean_params_t *mean_params,
    vx_size size_of_mean_param,
    vx_tensor outputs);

/*! \brief [Graph] Creates squeeze layer node.
* \details
*    Remove dimensions of size 1 from the input tensor.
*
* \param [in] graph The handle to the graph.
* \param [in] input A n-D tensor, specifying the tensor to be squeezed.
* \param [in] squeeze_params paraments <tt>\ref vx_nn_squeeze_params_t </tt>.
* \param [in] size_of_squeeze_param [static] The size of the vx_nn_squeeze_params_t.
* \param [out] output A n-D tensor of the same type as input. Contains the same data as input, 
*              but has one or more dimensions of size 1 removed.
* \return <tt> vx_node</tt>.
* \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_tensor
* \version 0.5
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorSqueezeNode(
    vx_graph graph,
    vx_tensor inputs,
    const vx_nn_squeeze_params_t *squeeze_params,
    vx_size size_of_squeeze_param,
    vx_tensor outputs);

/*! \brief [Graph] Creates stride slice layer node.
* \details
*    Extracts a stride slice of a tensor.
*
* \param [in] graph The handle to the graph.
* \param [in] input A n-D tensor, specifying the tensor to be sliced.
* \param [in] stride_slice_params paraments <tt>\ref vx_nn_stride_slice_params_t </tt>.
* \param [in] size_of_stride_slice_param [static] The size of the vx_nn_stride_slice_params_t.
* \param [out] output A n-D tensor of the same type as input.
* \return <tt> vx_node</tt>.
* \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
* successful creation should be checked using <tt>\ref vxGetStatus</tt>.
* \ingroup group_tensor
* \version 0.5
*/
VX_API_ENTRY vx_node VX_API_CALL vxTensorStrideSliceNode(
    vx_graph graph,
    vx_tensor inputs,
    const vx_nn_stride_slice_params_t *stride_slice_params,
    vx_size size_of_stride_slice_param,
    vx_tensor outputs);

/*! \brief Input parameters for query hardware caps.
 * \ingroup group_context
 */
typedef struct _vx_hardware_caps_params_t
{
    vx_uint32 ecoID;        /*!< \brief  hardware eco ID.*/
    vx_uint32 customerID;   /*!< \brief  hardware custmoer ID. ecoID and custmomerID can identify a unique hardware.*/
    vx_bool   evis1;         /*!< \brief  evs1 If true, hardware support evis1.*/
    vx_bool   evis2;         /*!< \brief  evs2 If true, hardware support evis2.*/
} vx_hardware_caps_params_t;

/*! \brief Input parameters for query hardware caps.
 * \ingroup group_context
 */
typedef struct _vx_hardware_caps_params_ext_t
{
    vx_hardware_caps_params_t base;
    vx_uint32 subGroupSize;        /*!< \brief  shader sub-group size.*/
} vx_hardware_caps_params_ext_t;

/*! \brief Queries hardware caps information.
 * \param [in] context The reference to the context.
 * \param [in] hardware_caps_params <tt>\ref vx_hardware_caps_params_t </tt>.
 * \param [in] size_of_hardware_caps_param [static] Size in bytes of hardware_caps_params.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE context is not a valid <tt>\ref vx_context</tt> reference.
 * \retval VX_ERROR_INVALID_PARAMETERS If any of the other parameters are incorrect.
 * \ingroup group_context
 */
VX_API_ENTRY vx_status VX_API_CALL vxQueryHardwareCaps(
    vx_context                          context,
    const vx_hardware_caps_params_t   * hardware_caps_params,
    vx_size                             size_of_hardware_caps_param
    );

#ifdef  __cplusplus
}
#endif


#endif
