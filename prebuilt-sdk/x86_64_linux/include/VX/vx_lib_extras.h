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

#ifndef _VX_EXT_EXTRAS_H_
#define _VX_EXT_EXTRAS_H_

/*! \file
 * \brief Extras Extension.
 *
 * \defgroup group_extras_ext Khronos Extras Extension.
 * \brief A Set of Kernels which extend OpenVX.
 *
 * \defgroup group_vision_function_laplacian_image Kernel: Laplacian Filter
 * \brief Computes a Laplacian filter over a window of the input image.
 * \details This filter uses the follow convolution matrix:
 \f[
 \mathbf{K}_{gaussian} = \begin{vmatrix}
   1 & 1 & 1\\
   1 &-8 & 1\\
   1 & 1 & 1
 \end{vmatrix} * \frac{1}{1}
 \f]
 *
 * \defgroup group_vision_function_scharr3x3 Kernel: Sobel 3x3
 * \brief The Scharr Image Filter Kernel
 * \details This kernel produces two output planes (one can be omitted)
 * in the x and y plane. The Scharr operators \f$G_x, G_y\f$ are defined as:
   \f[
        \mathbf{G}_x=\begin{vmatrix}
        -3 & 0 & +3\\
        -10& 0 & +10\\
        -3 & 0 & +3
        \end{vmatrix}
        ,
        \mathbf{G}_y=\begin{vmatrix}
        -3 & -10 & -3 \\
         0 &  0 & 0  \\
        +3 & +10 & +3
        \end{vmatrix}

    \f]
 *
 */

/*! \brief The Khronos Extras Library
 * \ingroup group_extras_ext
 */
#define VX_LIBRARY_KHR_EXTRAS (0xFE)

/*! \brief The Khronos Extras Kernels.
 * \ingroup group_extras_ext
 */
enum vx_kernel_extras_ext_e {
    /*! \brief The Non-Maximum Supression Kernel for Canny.
     * \note Use "org.khronos.extra.nonmaximasuppression" to \ref vxGetKernelByName.
     * \param [in] vx_image The magnitude image in VX_DF_IMAGE_U8.
     * \param [in] vx_image The phase image in VX_DF_IMAGE_U8.
     * \param [out] vx_image The edge image in VX_DF_IMAGE_U8.
     * \ingroup group_vision_function_nonmaxsuppression
     */
    VX_KERNEL_EXTRAS_NONMAXSUPPRESSION_CANNY = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x0,

    /*! \brief The laplacian filter kernel.
     * \note Use "org.khronos.extras.laplacian3x3" to \ref vxGetKernelByName.
     * \param [in] vx_image The VX_DF_IMAGE_U8 input image.
     * \param [out] vx_image The VX_DF_IMAGE_U8 output image.
     * \see group_vision_function_laplacian_image
     */
    VX_KERNEL_EXTRAS_LAPLACIAN_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x1,

    /*! \brief The scharr filter kernel.
     * \note Use "org.khronos.extras.scharr3x3" to \ref vxGetKernelByName.
     * \param [in] vx_image The VX_DF_IMAGE_U8 input image.
     * \param [out] vx_image The VX_DF_IMAGE_S16 output gradient x image.
     * \param [out] vx_image The VX_DF_IMAGE_S16 output gradient y image.
     * \see group_vision_function_scharr3x3
     */
    VX_KERNEL_EXTRAS_SCHARR_3x3 = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x2,

    /*! \brief The Harris Score Kernel.
     * \note use "org.khronos.extras.harris_score".
     * \param [in] vx_image A VX_DF_IMAGE_S16 X Gradient
     * \param [in] vx_image A VX_DF_IMAGE_S16 Y Gradient
     * \param [in] vx_scalar A block size.
     * \param [out] vx_image A VX_DF_IMAGE_S32 corner score per pixel.
     * \ingroup group_vision_function_harris_score
     */
    VX_KERNEL_EXTRAS_HARRIS_SCORE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x3,

    /*! \brief The Sobel MxN kernel.
     * \note Use "org.khronos.extras.sobelMxN" to \ref vxGetKernelByName.
     * \param [in] vx_image The VX_DF_IMAGE_U8 input image.
     * \param [in] vx_scalar Window Size (3,5,7)
     * \param [out] vx_image The VX_DF_IMAGE_S16 output gradient x image.
     * \param [out] vx_image The VX_DF_IMAGE_S16 output gradient y image.
     * \see group_vision_function_sobelmxn
     */
    VX_KERNEL_EXTRAS_SOBEL_MxN = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x4,

    /*! \brief The image to list converter.
     * \param [in] vx_image The VX_DF_IMAGE_U8 or VX_DF_IMAGE_S32 image.
     * \param [out] vx_array The array of output
     * \param [out] vx_scalar The total number of non zero points in image (optional)
     * \ingroup group_vision_function_image_lister
     */
    VX_KERNEL_EXTRAS_IMAGE_LISTER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x5,

    /*! \brief The Euclidean Non-Maximum Suppression Kernel for Harris Corners.
     * \param [in] vx_image The VX_DF_IMAGE_F32 image.
     * \param [in] vx_scalar The minimum threshold
     * \param [in] vx_scalar The euclidean distance from the considered pixel.
     * \param [out] vx_image The VX_DF_IMAGE_F32 image.
     * \ingroup group_vision_function_euclidean_nonmax
     */
    VX_KERNEL_EXTRAS_EUCLIDEAN_NONMAXSUPPRESSION_HARRIS = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x6,

    /*! \brief Elementwise binary norm kernel.
     * \param [in] vx_image Left image (VX_DF_IMAGE_S16).
     * \param [in] vx_image Right image (VX_DF_IMAGE_S16).
     * \param [in] vx_scalar Norm type (vx_norm_type_e).
     * \param [in] vx_image Output image (VX_DF_IMAGE_U16).
     */
    VX_KERNEL_EXTRAS_ELEMENTWISE_NORM = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x7,

    /*! \brief Edge tracing kernel.
     * \param [in] vx_image Norm image (VX_DF_IMAGE_U16).
     * \param [in] vx_image Phase image (VX_DF_IMAGE_U8).
     * \param [in] vx_threshold Threshold (VX_THRESHOLD_TYPE_RANGE).
     * \param [out] vx_image Output binary image (VX_DF_IMAGE_U8).
     */
    VX_KERNEL_EXTRAS_EDGE_TRACE = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_EXTRAS) + 0x8
};

/*! \brief Extra VX_DF_IMAGE codes supported by this extension. */
enum _vx_extra_df_image {
    /*! \brief A single plane of 32 bit float data.
     * The range of the data is not specified.
     */
    VX_DF_IMAGE_F32 = VX_DF_IMAGE('F','0','3','2'),
};

#ifdef  __cplusplus
extern "C" {
#endif

/*! \brief [Graph] Creates a Non Max Suppress Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_node vxNonMaxSuppressionCannyNode(vx_graph graph, vx_image mag, vx_image phase, vx_image edge);

/*! \brief [Immediate] Creates a Non Max Suppress Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_status vxuNonMaxSuppressionCanny(vx_context context, vx_image mag, vx_image phase, vx_image edge);

/*! \brief [Graph] Creates a Laplacian Filter Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_node vxLaplacian3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Immediate] Computes a laplacian filter on the image by a 3x3 window.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_status vxuLaplacian3x3(vx_context context, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Scharr Filter Node.
 * \param [in] graph The handle to the graph.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_node vxScharr3x3Node(vx_graph graph, vx_image input, vx_image output1, vx_image output2);

/*! \brief [Immediate] Computes a Scharr filter on the image by a 3x3 window.
 * \param [in] input The input image in VX_DF_IMAGE_U8 format.
 * \param [out] output The output image in VX_DF_IMAGE_U8 format.
 * \ingroup group_vision_function_laplacian_image
 */
vx_status vxuScharr3x3(vx_context context, vx_image input, vx_image output1, vx_image output2);

vx_node vxSobelMxNNode(vx_graph graph, vx_image input, vx_scalar win, vx_image gx, vx_image gy);

vx_status vxuSobelMxN(vx_context context, vx_image input, vx_scalar win, vx_image gx, vx_image gy);

vx_node vxHarrisScoreNode(vx_graph graph,
                         vx_image gx,
                         vx_image gy,
                         vx_scalar sensitivity,
                         vx_scalar grad_size,
                         vx_scalar block_size,
                         vx_scalar shift, 
                         vx_image score);

vx_status vxuHarrisScore(vx_context context, vx_image gx,
                         vx_image gy,
                         vx_scalar sensitivity,
                         vx_scalar grad_size,
                         vx_scalar block_size,
                         vx_scalar shift, 
                         vx_image score);

vx_node vxEuclideanNonMaxHarrisNode(vx_graph graph,
                              vx_image input,
                              vx_scalar strength_thresh,
                              vx_scalar min_distance,
                              vx_image output);

vx_status vxuEuclideanNonMaxHarris(vx_context context, vx_image input,
                             vx_scalar strength_thresh,
                             vx_scalar min_distance,
                             vx_image output);

vx_node vxImageListerNode(vx_graph graph, vx_image input, vx_array arr, vx_scalar num_points);

vx_status vxuImageLister(vx_context context, vx_image input,
                         vx_array arr, vx_scalar num_points);

vx_node vxElementwiseNormNode(vx_graph graph, vx_image input_x, vx_image input_y, vx_scalar norm_type, vx_image output);

vx_node vxEdgeTraceNode(vx_graph graph, vx_image norm, vx_threshold threshold, vx_image output);

#ifdef  __cplusplus
}
#endif

#endif  /* _VX_EXT_EXTRAS_H_ */

