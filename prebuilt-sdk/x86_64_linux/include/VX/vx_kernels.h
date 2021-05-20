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

#ifndef _OPENVX_KERNELS_H_
#define _OPENVX_KERNELS_H_

/*!
 * \file
 * \brief The list of supported kernels in the OpenVX standard.
 */

#ifdef  __cplusplus
extern "C" {
#endif

/*!
 * \brief The standard list of available libraries
 * \ingroup group_kernel
 */
enum vx_library_e {
    /*! \brief The base set of kernels as defined by Khronos. */
    VX_LIBRARY_KHR_BASE = 0x0,
};

/*!
 * \brief The standard list of available vision kernels.
 *
 * Each kernel listed here can be used with the <tt>\ref vxGetKernelByEnum</tt> call.
 * When programming the parameters, use
 * \arg <tt>\ref VX_INPUT</tt> for [in]
 * \arg <tt>\ref VX_OUTPUT</tt> for [out]
 * \arg <tt>\ref VX_BIDIRECTIONAL</tt> for [in,out]
 *
 * When programming the parameters, use
 * \arg <tt>\ref VX_TYPE_IMAGE</tt> for a <tt>\ref vx_image</tt> in the size field of <tt>\ref vxGetParameterByIndex</tt> or <tt>\ref vxSetParameterByIndex</tt>  * \arg <tt>\ref VX_TYPE_ARRAY</tt> for a <tt>\ref vx_array</tt> in the size field of <tt>\ref vxGetParameterByIndex</tt> or <tt>\ref vxSetParameterByIndex</tt>  * \arg or other appropriate types in \ref vx_type_e.
 * \ingroup group_kernel
 */
enum vx_kernel_e {

    /*!
     * \brief The Color Space conversion kernel.
     * \details The conversions are based on the <tt>\ref vx_df_image_e</tt> code in the images.
     * \see group_vision_function_colorconvert
     */
    VX_KERNEL_COLOR_CONVERT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1,

    /*!
     * \brief The Generic Channel Extraction Kernel.
     * \details This kernel can remove individual color channels from an interleaved
     * or semi-planar, planar, sub-sampled planar image. A client could extract
     * a red channel from an interleaved RGB image or do a Luma extract from a
     * YUV format.
     * \see group_vision_function_channelextract
     */
    VX_KERNEL_CHANNEL_EXTRACT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2,

    /*!
     * \brief The Generic Channel Combine Kernel.
     * \details This kernel combine multiple individual planes into a single
     * multiplanar image of the type specified in the output image.
     * \see group_vision_function_channelcombine
     */
    VX_KERNEL_CHANNEL_COMBINE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3,

    /*! \brief The Sobel 3x3 Filter Kernel.
     * \see group_vision_function_sobel3x3
     */
    VX_KERNEL_SOBEL_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x4,

    /*!
     * \brief The Magnitude Kernel.
     * \details This kernel produces a magnitude plane from two input gradients.
     * \see group_vision_function_magnitude
     */
    VX_KERNEL_MAGNITUDE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x5,

    /*!
     * \brief The Phase Kernel.
     * \details This kernel produces a phase plane from two input gradients.
     * \see group_vision_function_phase
     */
    VX_KERNEL_PHASE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x6,

    /*!
     * \brief The Scale Image Kernel.
     * \details This kernel provides resizing of an input image to an output image.
     * The scaling factor is determined but the relative sizes of the input and
     * output.
     * \see group_vision_function_scale_image
     */
    VX_KERNEL_SCALE_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x7,

    /*! \brief The Table Lookup kernel
     * \see group_vision_function_lut
     */
    VX_KERNEL_TABLE_LOOKUP = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x8,

    /*! \brief The Histogram Kernel.
     * \see group_vision_function_histogram
     */
    VX_KERNEL_HISTOGRAM = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x9,

    /*! \brief The Histogram Equalization Kernel.
     * \see group_vision_function_equalize_hist
     */
    VX_KERNEL_EQUALIZE_HISTOGRAM = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xA,

    /*! \brief The Absolute Difference Kernel.
     * \see group_vision_function_absdiff
     */
    VX_KERNEL_ABSDIFF = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xB,

    /*! \brief The Mean and Standard Deviation Kernel.
     * \see group_vision_function_meanstddev
     */
    VX_KERNEL_MEAN_STDDEV = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xC,

    /*! \brief The Threshold Kernel.
     * \see group_vision_function_threshold
     */
    VX_KERNEL_THRESHOLD = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xD,

    /*! \brief The Integral Image Kernel.
     * \see group_vision_function_integral_image
     */
    VX_KERNEL_INTEGRAL_IMAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xE,

    /*! \brief The dilate kernel.
     * \see group_vision_function_dilate_image
     */
    VX_KERNEL_DILATE_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0xF,

    /*! \brief The erode kernel.
     * \see group_vision_function_erode_image
     */
    VX_KERNEL_ERODE_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x10,

    /*! \brief The median image filter.
     * \see group_vision_function_median_image
     */
    VX_KERNEL_MEDIAN_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x11,

    /*! \brief The box filter kernel.
     * \see group_vision_function_box_image
     */
    VX_KERNEL_BOX_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x12,

    /*! \brief The gaussian filter kernel.
     * \see group_vision_function_gaussian_image
     */
    VX_KERNEL_GAUSSIAN_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x13,

    /*! \brief The custom convolution kernel.
     * \see group_vision_function_custom_convolution
     */
    VX_KERNEL_CUSTOM_CONVOLUTION = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x14,

    /*! \brief The gaussian image pyramid kernel.
     * \see group_vision_function_gaussian_pyramid
     */
    VX_KERNEL_GAUSSIAN_PYRAMID = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x15,

    /*! \brief The min and max location kernel.
     * \see group_vision_function_minmaxloc
     */
    VX_KERNEL_MINMAXLOC = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x19,

    /*! \brief The bit-depth conversion kernel.
     * \see group_vision_function_convertdepth
     */
    VX_KERNEL_CONVERTDEPTH = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1A,

    /*! \brief The Canny Edge Detector.
     * \see group_vision_function_canny
     */
    VX_KERNEL_CANNY_EDGE_DETECTOR = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1B,

    /*! \brief The Bitwise And Kernel.
     * \see group_vision_function_and
     */
    VX_KERNEL_AND = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1C,

    /*! \brief The Bitwise Inclusive Or Kernel.
     * \see group_vision_function_or
     */
    VX_KERNEL_OR = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1D,

    /*! \brief The Bitwise Exclusive Or Kernel.
     * \see group_vision_function_xor
     */
    VX_KERNEL_XOR = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1E,

    /*! \brief The Bitwise Not Kernel.
     * \see group_vision_function_not
     */
    VX_KERNEL_NOT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x1F,

    /*! \brief The Pixelwise Multiplication Kernel.
     * \see group_vision_function_mult
     */
    VX_KERNEL_MULTIPLY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x20,

    /*! \brief The Addition Kernel.
     * \see group_vision_function_add
     */
    VX_KERNEL_ADD = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x21,

    /*! \brief The Subtraction Kernel.
     * \see group_vision_function_sub
     */
    VX_KERNEL_SUBTRACT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x22,

    /*! \brief The Warp Affine Kernel.
     * \see group_vision_function_warp_affine
     */
    VX_KERNEL_WARP_AFFINE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x23,

    /*! \brief The Warp Perspective Kernel.
     * \see group_vision_function_warp_perspective
     */
    VX_KERNEL_WARP_PERSPECTIVE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x24,

    /*! \brief The Harris Corners Kernel.
     * \see group_vision_function_harris
     */
    VX_KERNEL_HARRIS_CORNERS = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x25,

    /*! \brief The FAST Corners Kernel.
     * \see group_vision_function_fast
     */
    VX_KERNEL_FAST_CORNERS = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x26,

    /*! \brief The Optical Flow Pyramid (LK) Kernel.
     * \see group_vision_function_opticalflowpyrlk
     */
    VX_KERNEL_OPTICAL_FLOW_PYR_LK = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x27,

    /*! \brief The Remap Kernel.
     * \see group_vision_function_remap
     */
    VX_KERNEL_REMAP = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x28,

    /*! \brief The Half Scale Gaussian Kernel.
     * \see group_vision_function_scale_image
     */
    VX_KERNEL_HALFSCALE_GAUSSIAN = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x29,

    VX_KERNEL_MAX_1_0, /*!< \internal Used for VX1.0 bounds checking in the conformance test. */

    /* kernel added in OpenVX 1.1 */

    /*! \brief The Laplacian Image Pyramid Kernel.
    * \see group_vision_function_laplacian_pyramid
    */
    VX_KERNEL_LAPLACIAN_PYRAMID = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2A,

    /*! \brief The Laplacian Pyramid Reconstruct Kernel.
    * \see group_vision_function_laplacian_pyramid
    */
    VX_KERNEL_LAPLACIAN_RECONSTRUCT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2B,

    /*! \brief The Non Linear Filter Kernel.
    * \see group_vision_function_nonlinear_filter
    */
    VX_KERNEL_NON_LINEAR_FILTER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2C,

    VX_KERNEL_MAX_1_1, /*!< \internal Used for VX1.1 bounds checking in the conformance test. */

    /* kernel added in OpenVX 1.2 */

    /*! \brief The Match Template Kernel.
    * \see group_vision_match_template
    */
    VX_KERNEL_MATCH_TEMPLATE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2D,

    /*! \brief The LBP Kernel.
    * \see group_lbp
    */
    VX_KERNEL_LBP = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2E,

    /*! \brief The hough lines probability Kernel.
    * \see group_vision_hough_lines_p
    */
    VX_KERNEL_HOUGH_LINES_P = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x2F,

    /*! \brief The tensor multiply Kernel.
    * \see group_vision_function_tensor_multiply
    */
    VX_KERNEL_TENSOR_MULTIPLY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x30,

    /*! \brief The tensor add Kernel.
    * \see group_vision_function_tensor_add
    */
    VX_KERNEL_TENSOR_ADD = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x31,

    /*! \brief The tensor subtract Kernel.
    * \see group_vision_function_tensor_subtract
    */
    VX_KERNEL_TENSOR_SUBTRACT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x32,

    /*! \brief The tensor table look up Kernel.
    * \see group_vision_function_tensor_tablelookup
    */
    VX_KERNEL_TENSOR_TABLE_LOOKUP = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x33,

    /*! \brief The tensor transpose Kernel.
    * \see group_vision_function_tensor_transpose
    */
    VX_KERNEL_TENSOR_TRANSPOSE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x34,

    /*! \brief The tensor convert depth Kernel.
    * \see group_vision_function_tensor_convert_depth
    */
    VX_KERNEL_TENSOR_CONVERT_DEPTH = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x35,

    /*! \brief The tensor matrix multiply Kernel.
    * \see group_vision_function_tensor_matrix_multiply
    */
    VX_KERNEL_TENSOR_MATRIX_MULTIPLY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x36,

    /*! \brief The data object copy kernel.
    * \see group_vision_function_copy
    */
    VX_KERNEL_COPY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x37,

    /*! \brief The non-max suppression kernel.
    * \see group_vision_function_nms
    */
    VX_KERNEL_NON_MAX_SUPPRESSION = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x38,

    /*! \brief The scalar operation kernel.
    * \see group_control_flow
    */
    VX_KERNEL_SCALAR_OPERATION = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x39,

    /*! \brief The  HOG features kernel.
    * \see group_vision_function_hog
    */
    VX_KERNEL_HOG_FEATURES = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3A,

    /*! \brief The HOG Cells kernel.
    * \see group_vision_function_hog
    */
    VX_KERNEL_HOG_CELLS = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3B,

    /*! \brief The bilateral filter kernel.
    * \see group_vision_function_bilateral_filter
    */
    VX_KERNEL_BILATERAL_FILTER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3C,

    /*! \brief The select kernel.
    * \see group_control_flow
    */
    VX_KERNEL_SELECT = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3D,

    /* insert new kernels here */

    /*! \brief The max kernel.
    * \see group_vision_function_max
    */
    VX_KERNEL_MAX = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3E,
    /*! \brief The min kernel.
    * \see group_vision_function_min
    */
    VX_KERNEL_MIN = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x3F,
 
    /*! \brief The weigthed average kernel.
     * \see group_vision_function_weighted_average
     */
    VX_KERNEL_WEIGHTED_AVERAGE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + 0x40,

    /* insert new kernels here */
    VX_KERNEL_NN_CONVOLUTION_RELU_POOLING_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x0,

    VX_KERNEL_NN_CONVOLUTION_RELU_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1,

    VX_KERNEL_NN_FULLY_CONNECTED_RELU_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2,

    //VX_KERNEL_NN_SOFTMAX_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x3,
    
    //VX_KERNEL_NN_NORMALIZATION_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x4, 

    VX_KERNEL_NN_LRN_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x3,

    //VX_KERNEL_NN_NORMALIZE_IMAGE_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x4, 

    //VX_KERNEL_NN_POOLING_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x7, 

    //VX_KERNEL_NN_ACTIVATION_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x9,

    VX_KERNEL_NN_LEAKY = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x4,

    VX_KERNEL_NN_BATCH_NORM = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x5,

    VX_KERNEL_NN_RPN = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x6,

    //VX_KERNEL_NN_ROIPOOL = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xD,

    VX_KERNEL_NN_CONCAT2_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x7,

    //VX_KERNEL_NN_CONVOLUTION_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xF,

    VX_KERNEL_NN_CONCATINDEFINITE_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x8,
    
    VX_KERNEL_NN_REORG_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x9,

    //VX_KERNEL_NN_DECONVOLUTION_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x12,

    VX_KERNEL_NN_TENSOR_DIV = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xA,

    VX_KERNEL_NN_L2NORMALIZE_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xB,

    VX_KERNEL_NN_TENSOR_COPY = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xC,

    VX_KERNEL_NN_CONVOLUTION_RELU_POOLING_LAYER2 = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xD,

    VX_KERNEL_NN_POOLING_LAYER2 = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xE,
    
    VX_KERNEL_NN_TENSOR_REDUCE_SUM = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0xF,
    
    VX_KERNEL_NN_TENSOR_PAD = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x10,

    VX_KERNEL_NN_LSTM_UNIT = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x11,

    VX_KERNEL_NN_LSTM_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x12,

    VX_KERNEL_NN_REORG2_LAYER         = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x13,
    
    VX_KERNEL_NN_TENSOR_ROUNDING      = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x14,
    
    VX_KERNEL_NN_HASH_LUT_LAYER       = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x15,
    
    VX_KERNEL_NN_LSH_PROJECTION_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x16,
    
    VX_KERNEL_NN_TENSOR_RESHPE        = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x17,
    
    VX_KERNEL_NN_LUT2_LAYER           = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x18,
    
    VX_KERNEL_NN_TENSOR_SCALE         = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x19,
    
    VX_KERNEL_NN_RNN_LAYER            = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1A,
    
    VX_KERNEL_NN_SOFTMAX2_LAYER       = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1B,
    
    VX_KERNEL_NN_SVDF_LAYER           = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1C,
    
    VX_KERNEL_NN_NORMALIZATION_LAYER2 = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1D,

    VX_KERNEL_NN_TENSOR_REVERSE       = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1E,

    VX_KERNEL_NN_TENSOR_TRANSPOSE     = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x1F,

    VX_KERNEL_NN_TENSOR_MEAN          = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x20,

    VX_KERNEL_NN_TENSOR_SQUEEZE       = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x21,

    VX_KERNEL_NN_TENSOR_STRIDE_SLICE  = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x22,

    VX_KERNEL_NN_TENSOR_PAD2          = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x23,

    VX_KERNEL_NN_YUV2RGB_SCALE        = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x24,

    VX_KERNEL_NN_PRELU                = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x25,

    VX_KERNEL_NN_GRU_UNIT_LAYER       = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x26,
    
    VX_KERNEL_NN_GRU_LAYER            = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x27,
    
    VX_KERNEL_NN_CONV_LSTM_UNIT_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x28,
    
    VX_KERNEL_NN_CONV_LSTM_LAYER      = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x29,

    VX_KERNEL_NN_FULLY_CONNECTED_LAYER = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2A,

    VX_KERNEL_NN_L2NORMALIZE_LAYER2    = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2B,

    VX_KERNEL_NN_CONVOLUTION_RELU_POOLING_ADD_LAYER2 = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2C,

    VX_KERNEL_NN_LUT_LAYER2           = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2D,

    VX_KERNEL_NN_CONVOLUTION_RELU_POOLING_MULTIPLY_LAYER2 = VX_KERNEL_BASE(VX_ID_VIVANTE, VX_LIBRARY_KHR_BASE) + 0x2E,

    VX_KERNEL_MAX_1_2, /*!< \internal Used for VX1.2 bounds checking in the conformance test. */
};

#ifdef  __cplusplus
}
#endif

#endif  /* _OPEN_VISION_LIBRARY_KERNELS_H_ */
