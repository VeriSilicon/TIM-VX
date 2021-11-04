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

#ifndef _OPENVX_NODES_H_
#define _OPENVX_NODES_H_

/*!
 * \file vx_nodes.h
 * \brief The "Simple" API interface for OpenVX. These APIs are just
 * wrappers around the more verbose functions defined in <tt>\ref vx_api.h</tt>.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief [Graph] Creates a color conversion node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image from which to convert.
 * \param [out] output The output image to which to convert, which must have the same dimensions as the input image.
 * \see <tt>VX_KERNEL_COLOR_CONVERT</tt>
 * \ingroup group_vision_function_colorconvert
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxColorConvertNode(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a channel extract node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image. Must be one of the defined \ref vx_df_image_e multi-channel formats.
 * \param [in] channel The <tt>\ref vx_channel_e</tt> channel to extract.
 * \param [out] output The output image. Must be <tt>\ref VX_DF_IMAGE_U8</tt>, and must have the same dimensions as the input image.
 * <tt>\see VX_KERNEL_CHANNEL_EXTRACT</tt>
 * \ingroup group_vision_function_channelextract
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxChannelExtractNode(vx_graph graph,
                             vx_image input,
                             vx_enum channel,
                             vx_image output);

/*! \brief [Graph] Creates a channel combine node.
 * \param [in] graph The graph reference.
 * \param [in] plane0 The plane that forms channel 0. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] plane1 The plane that forms channel 1. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] plane2 [optional] The plane that forms channel 2. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] plane3 [optional] The plane that forms channel 3. Must be <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] output The output image. The format of the image must be defined, even if the image is virtual. Must have the same dimensions as the input images
 * \see <tt>VX_KERNEL_CHANNEL_COMBINE</tt>
 * \ingroup group_vision_function_channelcombine
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxChannelCombineNode(vx_graph graph,
                             vx_image plane0,
                             vx_image plane1,
                             vx_image plane2,
                             vx_image plane3,
                             vx_image output);

/*! \brief [Graph] Creates a Phase node.
 * \param [in] graph The reference to the graph.
 * \param [in] grad_x The input x image. This must be in <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] grad_y The input y image. This must be in <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] orientation The phase image. This is in <tt>\ref VX_DF_IMAGE_U8</tt> format, and must have the same dimensions as the input images.
 * \see <tt>VX_KERNEL_PHASE</tt>
 * \ingroup group_vision_function_phase
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxPhaseNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image orientation);

/*! \brief [Graph] Creates a Sobel3x3 node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output_x [optional] The output gradient in the x direction in <tt>\ref VX_DF_IMAGE_S16</tt>. Must have the same dimensions as the input image.
 * \param [out] output_y [optional] The output gradient in the y direction in <tt>\ref VX_DF_IMAGE_S16</tt>. Must have the same dimensions as the input image.
 * \see <tt>VX_KERNEL_SOBEL_3x3</tt>
 * \ingroup group_vision_function_sobel3x3
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxSobel3x3Node(vx_graph graph, vx_image input, vx_image output_x, vx_image output_y);


/*! \brief [Graph] Create a Magnitude node.
 * \param [in] graph The reference to the graph.
 * \param [in] grad_x The input x image. This must be in <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] grad_y The input y image. This must be in <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] mag The magnitude image. This is in <tt>\ref VX_DF_IMAGE_S16</tt> format. Must have the same dimensions as the input image.
 * \see <tt>VX_KERNEL_MAGNITUDE</tt>
 * \ingroup group_vision_function_magnitude
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMagnitudeNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image mag);

/*! \brief [Graph] Creates a Scale Image Node.
 * \param [in] graph The reference to the graph.
 * \param [in] src The source image of type <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] dst The destination image of type <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] type The interpolation type to use. \see vx_interpolation_type_e.
 * \ingroup group_vision_function_scale_image
 * \note The destination image must have a defined size and format. The border modes
 *  <tt>\ref VX_NODE_BORDER</tt> value <tt>\ref VX_BORDER_UNDEFINED</tt>,
 *  <tt>\ref VX_BORDER_REPLICATE</tt> and <tt>\ref VX_BORDER_CONSTANT</tt> are supported.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxScaleImageNode(vx_graph graph, vx_image src, vx_image dst, vx_enum type);

/*! \brief [Graph] Creates a Table Lookup node. If a value from the input image is not present in the lookup table, the result is undefined.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] lut The LUT which is of type <tt>\ref VX_TYPE_UINT8</tt> if input image is <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_TYPE_INT16</tt> if input image is <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [out] output The output image of the same type and size as the input image.
 * \ingroup group_vision_function_lut
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTableLookupNode(vx_graph graph, vx_image input, vx_lut lut, vx_image output);

/*! \brief [Graph] Creates a Histogram node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] distribution The output distribution.
 * \ingroup group_vision_function_histogram
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxHistogramNode(vx_graph graph, vx_image input, vx_distribution distribution);

/*! \brief [Graph] Creates a Histogram Equalization node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The grayscale input image in <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [out] output The grayscale output image of type <tt>\ref VX_DF_IMAGE_U8</tt> with equalized brightness and contrast and same size as the input image.
 * \ingroup group_vision_function_equalize_hist
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxEqualizeHistNode(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates an AbsDiff node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 An input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] in2 An input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] out The output image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_absdiff
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxAbsDiffNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a mean value and optionally, a standard deviation node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image. <tt>\ref VX_DF_IMAGE_U8</tt> is supported.
 * \param [out] mean The <tt>\ref VX_TYPE_FLOAT32</tt> average pixel value.
 * \param [out] stddev [optional] The <tt>\ref VX_TYPE_FLOAT32</tt> standard deviation of the pixel values.
 * \ingroup group_vision_function_meanstddev
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMeanStdDevNode(vx_graph graph, vx_image input, vx_scalar mean, vx_scalar stddev);

/*! \brief [Graph] Creates a Threshold node and returns a reference to it.
 * \param [in] graph The reference to the graph in which the node is created.
 * \param [in] input The input image. Only images with format <tt>\ref VX_DF_IMAGE_U8</tt>
 * and <tt>\ref VX_DF_IMAGE_S16</tt> are supported.
 * \param [in] thresh The thresholding object that defines the parameters of
 * the operation. The <tt>\ref VX_THRESHOLD_INPUT_FORMAT</tt> must be the same as the input image format and
 * the <tt>\ref VX_THRESHOLD_OUTPUT_FORMAT</tt> must be the same as the output image format.
 * \param [out] output The output image, that will contain as pixel value
 * true and false values defined by \p thresh. Only images with format
 * <tt>\ref VX_DF_IMAGE_U8</tt> are supported. The dimensions are the same as the input image.
 * \ingroup group_vision_function_threshold
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation
 * should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxThresholdNode(vx_graph graph, vx_image input, vx_threshold thresh, vx_image output);

/*! \brief [Graph] Creates a Non-Maxima Suppression node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] mask [optional] Constrict suppression to a ROI. The mask image is of type <tt>\ref VX_DF_IMAGE_U8</tt> and must be the same dimensions as the input image.
 * \param [in] win_size The size of window over which to perform the localized non-maxima suppression. Must be odd, and less than or equal to the smallest dimension of the input image.
 * \param [out] output The output image, of the same type and size as the input, that has been non-maxima suppressed. 
 * \ingroup group_vision_function_nms
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxNonMaxSuppressionNode(vx_graph graph, vx_image input, vx_image mask, vx_int32 win_size, vx_image output);

/*! \brief [Graph] Creates an Integral Image Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U32</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_integral_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxIntegralImageNode(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates an Erosion Image Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_erode_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxErode3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Dilation Image Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_dilate_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxDilate3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Median Image Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_median_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMedian3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Box Filter Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_box_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxBox3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Gaussian Filter Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_gaussian_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxGaussian3x3Node(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a Non-linear Filter Node.
 * \param [in] graph The reference to the graph.
 * \param [in] function The non-linear filter function. See <tt>\ref vx_non_linear_filter_e</tt>.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] mask The mask to be applied to the Non-linear function. <tt>\ref VX_MATRIX_ORIGIN</tt> attribute is used
 *  to place the mask appropriately when computing the resulting image. See <tt>\ref vxCreateMatrixFromPattern</tt>.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format, which must have the same dimensions as the input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_nonlinear_filter
 */
VX_API_ENTRY vx_node VX_API_CALL vxNonLinearFilterNode(vx_graph graph, vx_enum function, vx_image input, vx_matrix mask, vx_image output);

/*! \brief [Graph] Creates a custom convolution node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [in] conv The <tt>\ref vx_int16</tt> convolution matrix.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_custom_convolution
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolveNode(vx_graph graph, vx_image input, vx_convolution conv, vx_image output);

/*! \brief [Graph] Creates a node for a Gaussian Image Pyramid.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> format.
 * \param [out] gaussian The Gaussian pyramid with <tt>\ref VX_DF_IMAGE_U8</tt> to construct.
 * \ingroup group_vision_function_gaussian_pyramid
 * \see group_pyramid
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxGaussianPyramidNode(vx_graph graph, vx_image input, vx_pyramid gaussian);

/*! \brief [Graph] Creates a node for a Laplacian Image Pyramid.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] laplacian The Laplacian pyramid with <tt>\ref VX_DF_IMAGE_S16</tt> to construct.
 * \param [out] output The lowest resolution image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format necessary to reconstruct the input image from the pyramid. The output image format should be same as input image format.
 * \ingroup group_vision_function_laplacian_pyramid
 * \see group_pyramid
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxLaplacianPyramidNode(vx_graph graph, vx_image input,
                                   vx_pyramid laplacian, vx_image output);

/*! \brief [Graph] Reconstructs an image from a Laplacian Image pyramid.
 * \param [in] graph The reference to the graph.
 * \param [in] laplacian The Laplacian pyramid with <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [in] input The lowest resolution image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format for the Laplacian pyramid.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format with the highest possible resolution reconstructed from the Laplacian pyramid. The output image format should be same as input image format.
 * \ingroup group_vision_function_laplacian_reconstruct
 * \see group_pyramid
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 */
VX_API_ENTRY vx_node VX_API_CALL vxLaplacianReconstructNode(vx_graph graph, vx_pyramid laplacian, vx_image input,
                                       vx_image output);
/*! \brief [Graph] Creates a image weighted average node.
 * \param [in] graph The reference to the graph.
 * \param [in] img1 The first input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] alpha The input <tt>\ref VX_TYPE_FLOAT32</tt> scalar value with a value in the range of \f$ 0.0 \le \alpha \le 1.0 \f$.
 * \param [in] img2 The second <tt>\ref VX_DF_IMAGE_U8</tt> image, which must have the same dimensions as the img1.
 * \param [out] output The output <tt>\ref VX_DF_IMAGE_U8</tt> image, which must have the same dimensions as the img1.
 * \ingroup group_vision_function_weighted_average
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxWeightedAverageNode(vx_graph graph, vx_image img1, vx_scalar alpha, vx_image img2, vx_image output);
/*! \brief [Graph] Creates a min,max,loc node.
 * \param [in] graph The reference to create the graph.
 * \param [in] input The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> format.
 * \param [out] minVal The minimum value in the image, which corresponds to the type of the input.
 * \param [out] maxVal The maximum value in the image, which corresponds to the type of the input.
 * \param [out] minLoc [optional] The minimum <tt>\ref VX_TYPE_COORDINATES2D</tt> locations. If the input image has several minimums, the kernel will return up to the capacity of the array.
 * \param [out] maxLoc [optional] The maximum <tt>\ref VX_TYPE_COORDINATES2D</tt> locations. If the input image has several maximums, the kernel will return up to the capacity of the array.
 * \param [out] minCount [optional] The total number of detected minimums in image. Use a <tt>\ref VX_TYPE_SIZE</tt> scalar.
 * \param [out] maxCount [optional] The total number of detected maximums in image. Use a <tt>\ref VX_TYPE_SIZE</tt> scalar.
 * \ingroup group_vision_function_minmaxloc
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMinMaxLocNode(vx_graph graph,
                        vx_image input,
                        vx_scalar minVal, vx_scalar maxVal,
                        vx_array minLoc, vx_array maxLoc,
                        vx_scalar minCount, vx_scalar maxCount);

/*! \brief [Graph] Creates a pixel-wise minimum kernel.
 * \param [in] graph The reference to the graph where to create the node.
 * \param [in] in1 The first input image. Must be of type <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] in2 The second input image. Must be of type <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [out] out The output image which will hold the result of min and will have the same type and dimensions of the imput images.
 * \ingroup group_vision_function_min
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMinNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a pixel-wise maximum kernel.
 * \param [in] graph The reference to the graph where to create the node.
 * \param [in] in1 The first input image. Must be of type <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] in2 The second input image. Must be of type <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [out] out The output image which will hold the result of max and will have the same type and dimensions of the imput images.
 * \ingroup group_vision_function_max
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMaxNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a bitwise AND node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [in] in2 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [out] out The <tt>\ref VX_DF_IMAGE_U8</tt> output image, which must have the same dimensions as the input images.
 * \ingroup group_vision_function_and
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxAndNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a bitwise INCLUSIVE OR node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [in] in2 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [out] out The <tt>\ref VX_DF_IMAGE_U8</tt> output image, which must have the same dimensions as the input images.
 * \ingroup group_vision_function_or
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxOrNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a bitwise EXCLUSIVE OR node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [in] in2 A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [out] out The <tt>\ref VX_DF_IMAGE_U8</tt> output image, which must have the same dimensions as the input images.
 * \ingroup group_vision_function_xor
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxXorNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out);

/*! \brief [Graph] Creates a bitwise NOT node.
 * \param [in] graph The reference to the graph.
 * \param [in] input A <tt>\ref VX_DF_IMAGE_U8</tt> input image.
 * \param [out] output The <tt>\ref VX_DF_IMAGE_U8</tt> output image, which must have the same dimensions as the input image.
 * \ingroup group_vision_function_not
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxNotNode(vx_graph graph, vx_image input, vx_image output);

/*! \brief [Graph] Creates a scalar operation node.
 * \param [in] graph The reference to the graph.
 * \param [in] scalar_operation A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_scalar_operation_e</tt> enumeration.
 * \param [in] a First scalar operand.
 * \param [in] b Second scalar operand.
 * \param [out] output Result of the scalar operation.
 * \ingroup group_control_flow
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxScalarOperationNode(vx_graph graph, vx_enum scalar_operation, vx_scalar a, vx_scalar b, vx_scalar output);

/*! \brief [Graph] Selects one of two data objects depending on the the value of a condition (boolean scalar), and copies its data into another data object.
 * \details This node supports predicated execution flow within a graph. All the data objects passed to this kernel shall
 * have the same object type and meta data. It is important to note that an implementation may optimize away the select and copy when virtual data
 * objects are used.\n
 * If there is a kernel node that contribute only into virtual data objects during the graph execution due to certain data path being eliminated by not
 * taken argument of select node, then the OpenVX implementation guarantees that there will not be any side effects to graph execution and node state.\n
 * If the path to a select node contains non-virtual objects, user nodes, or  nodes with completion callbacks, then that path may not be "optimized out"
 * because the callback must be executed and the non-virtual objects must be modified.
 * \param [in] graph The reference to the graph.
 * \param [in] condition <tt>\ref VX_TYPE_BOOL</tt> predicate variable.
 * \param [in] true_value Data object for true.
 * \param [in] false_value Data object for false.
 * \param [out] output Output data object.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_control_flow
  */
VX_API_ENTRY vx_node VX_API_CALL vxSelectNode(vx_graph graph, vx_scalar condition, vx_reference true_value, vx_reference false_value, vx_reference output);

/*! \brief [Graph] Creates an pixelwise-multiplication node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] in2 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] scale A non-negative <tt>\ref VX_TYPE_FLOAT32</tt> multiplied to each product before overflow handling.
 * \param [in] overflow_policy A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [in] rounding_policy A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_round_policy_e</tt> enumeration.
 * \param [out] out The output image, a <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> image. Must have the same type and dimensions of the imput images.
 * \ingroup group_vision_function_mult
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxMultiplyNode(vx_graph graph,
                       vx_image in1, vx_image in2,
                       vx_scalar scale,
                       vx_enum overflow_policy,
                       vx_enum rounding_policy,
                       vx_image out);

/*! \brief [Graph] Creates an arithmetic addition node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] in2 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>.
 * \param [in] policy A <tt>\ref VX_TYPE_ENUM</tt> of the \ref vx_convert_policy_e enumeration.
 * \param [out] out The output image, a <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> image, which must have the same dimensions as the input images.
 * \ingroup group_vision_function_add
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxAddNode(vx_graph graph,
                  vx_image in1, vx_image in2,
                  vx_enum policy,
                  vx_image out);

/*! \brief [Graph] Creates an arithmetic subtraction node.
 * \param [in] graph The reference to the graph.
 * \param [in] in1 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>, the minuend.
 * \param [in] in2 An input image, <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>, the subtrahend.
 * \param [in] policy A <tt>\ref VX_TYPE_ENUM</tt> of the \ref vx_convert_policy_e enumeration.
 * \param [out] out The output image, a <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt> image, which must have the same dimensions as the input images.
 * \ingroup group_vision_function_sub
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxSubtractNode(vx_graph graph,
                       vx_image in1, vx_image in2,
                       vx_enum policy,
                       vx_image out);

/*! \brief [Graph] Creates a bit-depth conversion node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image.
 * \param [out] output The output image with the same dimensions of the input image.
 * \param [in] policy A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [in] shift A scalar containing a <tt>\ref VX_TYPE_INT32</tt> of the shift value.
 * \ingroup group_vision_function_convertdepth
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvertDepthNode(vx_graph graph, vx_image input, vx_image output, vx_enum policy, vx_scalar shift);

/*! \brief [Graph] Creates a Canny Edge Detection Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] hyst The double threshold for hysteresis. The <tt>\ref VX_THRESHOLD_INPUT_FORMAT</tt> shall be either 
 * <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_S16</tt>. The <tt>\ref VX_THRESHOLD_OUTPUT_FORMAT</tt> is ignored.
 * \param [in] gradient_size The size of the Sobel filter window, must support at least 3, 5, and 7.
 * \param [in] norm_type A flag indicating the norm used to compute the gradient, <tt>\ref VX_NORM_L1</tt> or <tt>\ref VX_NORM_L2</tt>.
 * \param [out] output The output image in <tt>\ref VX_DF_IMAGE_U8</tt> format with values either 0 or 255.
 * \ingroup group_vision_function_canny
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxCannyEdgeDetectorNode(vx_graph graph, vx_image input, vx_threshold hyst,
                                vx_int32 gradient_size, vx_enum norm_type,
                                vx_image output);

/*! \brief [Graph] Creates an Affine Warp Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] matrix The affine matrix. Must be 2x3 of type \ref VX_TYPE_FLOAT32.
 * \param [in] type The interpolation type from <tt>\ref vx_interpolation_type_e</tt>.
 * <tt>\ref VX_INTERPOLATION_AREA</tt> is not supported.
 * \param [out] output The output <tt>\ref VX_DF_IMAGE_U8</tt> image and the same dimensions as the input image.
 * \ingroup group_vision_function_warp_affine
 * \note The border modes <tt>\ref VX_NODE_BORDER</tt> value <tt>\ref VX_BORDER_UNDEFINED</tt> and
 * <tt>\ref VX_BORDER_CONSTANT</tt> are supported.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxWarpAffineNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output);

/*! \brief [Graph] Creates a Perspective Warp Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] matrix The perspective matrix. Must be 3x3 of type <tt>\ref VX_TYPE_FLOAT32</tt>.
 * \param [in] type The interpolation type from <tt>\ref vx_interpolation_type_e</tt>.
 * <tt>\ref VX_INTERPOLATION_AREA</tt> is not supported.
 * \param [out] output The output <tt>\ref VX_DF_IMAGE_U8</tt> image with the same dimensions as the input image.
 * \ingroup group_vision_function_warp_perspective
 * \note The border modes <tt>\ref VX_NODE_BORDER</tt> value <tt>\ref VX_BORDER_UNDEFINED</tt> and
 * <tt>\ref VX_BORDER_CONSTANT</tt> are supported.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxWarpPerspectiveNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output);

/*! \brief [Graph] Creates a Harris Corners Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] strength_thresh The <tt>\ref VX_TYPE_FLOAT32</tt> minimum threshold with which to eliminate Harris Corner scores (computed using the normalized Sobel kernel).
 * \param [in] min_distance The <tt>\ref VX_TYPE_FLOAT32</tt> radial Euclidean distance for non-maximum suppression.
 * \param [in] sensitivity The <tt>\ref VX_TYPE_FLOAT32</tt> scalar sensitivity threshold \f$ k \f$ from the Harris-Stephens equation.
 * \param [in] gradient_size The gradient window size to use on the input. The
 * implementation must support at least 3, 5, and 7.
 * \param [in] block_size The block window size used to compute the Harris Corner score.
 * The implementation must support at least 3, 5, and 7.
 * \param [out] corners The array of <tt>\ref VX_TYPE_KEYPOINT</tt> objects. The order of the keypoints in this array is implementation dependent.
 * \param [out] num_corners [optional] The total number of detected corners in image. Use a \ref VX_TYPE_SIZE scalar.
 * \ingroup group_vision_function_harris
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxHarrisCornersNode(vx_graph graph,
                            vx_image input,
                            vx_scalar strength_thresh,
                            vx_scalar min_distance,
                            vx_scalar sensitivity,
                            vx_int32 gradient_size,
                            vx_int32 block_size,
                            vx_array corners,
                            vx_scalar num_corners);

/*! \brief [Graph] Creates a FAST Corners Node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] strength_thresh Threshold on difference between intensity of the central pixel and pixels on Bresenham's circle
 * of radius 3 (<tt>\ref VX_TYPE_FLOAT32</tt> scalar), with a value in the range of 0.0 \f$\le\f$ strength_thresh < 256.0.
 *  Any fractional value will be truncated to an integer.
 * \param [in] nonmax_suppression If true, non-maximum suppression is applied to
 * detected corners before being placed in the <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt> objects.
 * \param [out] corners Output corner <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>. The order of the
 *                      keypoints in this array is implementation dependent.
 * \param [out] num_corners [optional] The total number of detected corners in image. Use a \ref VX_TYPE_SIZE scalar.
 * \ingroup group_vision_function_fast
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxFastCornersNode(vx_graph graph, vx_image input, vx_scalar strength_thresh, vx_bool nonmax_suppression, vx_array corners, vx_scalar num_corners);

/*! \brief [Graph] Creates a Lucas Kanade Tracking Node.
 * \param [in] graph The reference to the graph.
 * \param [in] old_images Input of first (old) image pyramid in <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] new_images Input of destination (new) image pyramid <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] old_points An array of key points in a <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>; those key points are defined at
 *  the \a old_images high resolution pyramid.
 * \param [in] new_points_estimates An array of estimation on what is the output key points in a <tt>\ref vx_array</tt> of
 *  <tt>\ref VX_TYPE_KEYPOINT</tt>; those keypoints are defined at the \a new_images high resolution pyramid.
 * \param [out] new_points An output array of key points in a <tt>\ref vx_array</tt> of <tt>\ref VX_TYPE_KEYPOINT</tt>; those key points are
 *  defined at the \a new_images high resolution pyramid.
 * \param [in] termination The termination can be <tt>\ref VX_TERM_CRITERIA_ITERATIONS</tt> or <tt>\ref VX_TERM_CRITERIA_EPSILON</tt> or
 * <tt>\ref VX_TERM_CRITERIA_BOTH</tt>.
 * \param [in] epsilon The <tt>\ref vx_float32</tt> error for terminating the algorithm.
 * \param [in] num_iterations The number of iterations. Use a <tt>\ref VX_TYPE_UINT32</tt> scalar.
 * \param [in] use_initial_estimate Use a <tt>\ref VX_TYPE_BOOL</tt> scalar.
 * \param [in] window_dimension The size of the window on which to perform the algorithm. See
 *  <tt>\ref VX_CONTEXT_OPTICAL_FLOW_MAX_WINDOW_DIMENSION</tt>
 * \ingroup group_vision_function_opticalflowpyrlk
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxOpticalFlowPyrLKNode(vx_graph graph,
                               vx_pyramid old_images,
                               vx_pyramid new_images,
                               vx_array old_points,
                               vx_array new_points_estimates,
                               vx_array new_points,
                               vx_enum termination,
                               vx_scalar epsilon,
                               vx_scalar num_iterations,
                               vx_scalar use_initial_estimate,
                               vx_size window_dimension);

/*! \brief [Graph] Creates a Remap Node.
 * \param [in] graph The reference to the graph that will contain the node.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] table The remap table object.
 * \param [in] policy An interpolation type from <tt>\ref vx_interpolation_type_e</tt>.
 * <tt>\ref VX_INTERPOLATION_AREA</tt> is not supported.
 * \param [out] output The output <tt>\ref VX_DF_IMAGE_U8</tt> image with the same dimensions as the input image.
 * \note The border modes <tt>\ref VX_NODE_BORDER</tt> value <tt>\ref VX_BORDER_UNDEFINED</tt> and
 * <tt>\ref VX_BORDER_CONSTANT</tt> are supported.
 * \return A <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_remap
 */
VX_API_ENTRY vx_node VX_API_CALL vxRemapNode(vx_graph graph,
                    vx_image input,
                    vx_remap table,
                    vx_enum policy,
                    vx_image output);

/*! \brief [Graph] Performs a Gaussian Blur on an image then half-scales it. The interpolation mode used is nearest-neighbor.
 * \details The output image size is determined by:
 * \f[
 * W_{output} = \frac{W_{input} + 1}{2} \\
 * ,
 * H_{output} = \frac{H_{input} + 1}{2}
 * \f]
 * \param [in] graph The reference to the graph.
 * \param [in] input The input <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [out] output The output <tt>\ref VX_DF_IMAGE_U8</tt> image.
 * \param [in] kernel_size The input size of the Gaussian filter. Supported values are 1, 3 and 5.
 * \ingroup group_vision_function_scale_image
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxHalfScaleGaussianNode(vx_graph graph, vx_image input, vx_image output, vx_int32 kernel_size);

VX_API_ENTRY vx_node VX_API_CALL vxCensus3x3Node(vx_graph graph, vx_image src, vx_image dst);

/*! \brief [Graph]  The Node Compares an image template against overlapped image regions.
 * \details The detailed equation to the matching can be found in <tt>\ref vx_comp_metric_e</tt>.
 * The output of the template matching node is a comparison map as described in <tt>\ref vx_comp_metric_e</tt>.
 * The Node have a limitation on the template image size (width*height). It should not be larger then 65535.
 * If the valid region of the template image is smaller than the entire template image, the result in the destination image is implementation-dependent.
 * \param [in] graph The reference to the graph.
 * \param [in] src The input image of type <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] templateImage Searched template of type <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] matchingMethod attribute specifying the comparison method <tt>\ref vx_comp_metric_e</tt>. This function support only <tt>\ref VX_COMPARE_CCORR_NORM</tt> and <tt>\ref VX_COMPARE_L2</tt>.
 * \param [out] output Map of comparison results. The output is an image of type VX_DF_IMAGE_S16
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_match_template
 */
 VX_API_ENTRY vx_node VX_API_CALL vxMatchTemplateNode(vx_graph graph, vx_image src, vx_image templateImage, vx_enum matchingMethod, vx_image output);

 /*! \brief [Graph] Creates a node that extracts LBP image from an input image
* \param [in] graph    The reference to the graph.
* \param [in] in        An input image in <tt>vx_image</tt>. Or \f$ SrcImg\f$ in the equations. the image is of type <tt>\ref VX_DF_IMAGE_U8</tt>
* \param [in] format    A variation of LBP like original LBP and mLBP. see <tt> \ref vx_lbp_format_e </tt>
* \param [in] kernel_size Kernel size. Only size of 3 and 5 are supported
* \param [out] out    An output image in <tt>vx_image</tt>.Or \f$ DstImg\f$ in the equations. the image is of type <tt>\ref VX_DF_IMAGE_U8</tt> with the same dimensions as the input image.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
* \ingroup group_vision_function_lbp
*/
VX_API_ENTRY vx_node VX_API_CALL vxLBPNode(vx_graph graph, vx_image in, vx_enum format, vx_int8 kernel_size, vx_image out);

/*! \brief [Graph] Performs cell calculations for the average gradient magnitude and gradient orientation histograms.
 * \details Firstly, the gradient magnitude and gradient orientation are computed for each pixel in the input image.
 * Two 1-D centred, point discrete derivative masks are applied to the input image in the horizontal and vertical directions.
 * \f[ M_h = [-1, 0, 1] \f] and \f[ M_v = [-1, 0, 1]^T \f]
 * \f$G_v\f$ is the result of applying mask \f$M_v\f$ to the input image, and \f$G_h\f$ is the result of applying mask \f$M_h\f$ to the input image.
 * The border mode used for the gradient calculation is implementation dependent. Its behavior should be similar to <tt>\ref VX_BORDER_UNDEFINED</tt>.
 * The gradient magnitudes and gradient orientations for each pixel are then calculated in the following manner.
 * \f[ G(x,y) = \sqrt{G_v(x,y)^2 + G_h(x,y)^2} \f]
 * \f[ \theta(x,y) = arctan(G_v(x,y), G_h(x,y)) \f]
 * where \f$arctan(v, h)\f$
 * is \f$ tan^{-1}(v/h)\f$ when \f$h!=0\f$,
 *
 * \f$ -pi/2 \f$ if \f$v<0\f$ and \f$h==0\f$,
 *
 * \f$  pi/2  \f$ if \f$v>0\f$ and \f$h==0\f$
 *
 * and \f$     0  \f$ if \f$v==0\f$ and \f$h==0\f$
 *
 * Secondly, the gradient magnitudes and orientations are used to compute the bins output tensor and optional magnitudes output tensor.
 * These tensors are computed on a cell level where the cells are rectangular in shape.
 * The magnitudes tensor contains the average gradient magnitude for each cell.
 * \f[magnitudes(c) = \frac{1}{(cell\_width * cell\_height)}\sum\limits_{w=0}^{cell\_width} \sum\limits_{h=0}^{cell\_height} G_c(w,h)\f]
 * where \f$G_c\f$ is the gradient magnitudes related to cell \f$c\f$.
 * The bins tensor contains histograms of gradient orientations for each cell.
 * The gradient orientations at each pixel range from 0 to 360 degrees.  These are quantised into a set of histogram bins based on the num_bins parameter.
 * Each pixel votes for a specific cell histogram bin based on its gradient orientation.  The vote itself is the pixel's gradient magnitude.
 * \f[bins(c, n) = \sum\limits_{w=0}^{cell\_width} \sum\limits_{h=0}^{cell\_height} G_c(w,h) * 1[B_c(w, h, num\_bins) == n]\f]
 * where \f$B_c\f$ produces the histogram bin number based on the gradient orientation of the pixel at location (\f$w\f$, \f$h\f$) in cell \f$c\f$ based on
 * the \f$num\_bins\f$ and \f[1[B_c(w, h, num\_bins) == n]\f] is a delta-function with value 1 when \f$B_c(w, h, num\_bins) == n\f$ or 0 otherwise. 
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image of type <tt>\ref VX_DF_IMAGE_U8</tt>.
 * \param [in] cell_width The histogram cell width of type <tt>\ref VX_TYPE_INT32</tt>.
 * \param [in] cell_height The histogram cell height of type <tt>\ref VX_TYPE_INT32</tt>.
 * \param [in] num_bins  The histogram size of type <tt>\ref VX_TYPE_INT32</tt>.
 * \param [out] magnitudes (Optional) The output average gradient magnitudes per cell of <tt>\ref vx_tensor</tt> of type <tt>\ref VX_TYPE_INT16</tt> of size \f$ [floor(image_{width}/cell_{width}) ,floor(image_{height}/cell_{height}) ] \f$.
 * \param [out] bins       The output gradient orientation histograms per cell of <tt>\ref vx_tensor</tt> of type <tt>\ref VX_TYPE_INT16</tt> of size \f$ [floor(image_{width}/cell_{width}) ,floor(image_{height}/cell_{height}), num_{bins}] \f$.
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_vision_function_hog
 */
VX_API_ENTRY vx_node VX_API_CALL vxHOGCellsNode(vx_graph graph, vx_image input, vx_int32 cell_width, vx_int32 cell_height, vx_int32 num_bins, vx_tensor magnitudes, vx_tensor bins);

/*! \brief [Graph] The node produces HOG features for the W1xW2 window in a sliding window fashion over the whole input image. Each position produces a HOG feature vector.
 * \details Firstly if a magnitudes tensor is provided the cell histograms in the bins tensor are normalised by the average cell gradient magnitudes.
 \f[bins(c,n) = \frac{bins(c,n)}{magnitudes(c)}\f]
 * To account for changes in illumination and contrast the cell histograms must be locally normalized which requires grouping the cell histograms together into larger spatially connected blocks.
 * Blocks are rectangular grids represented by three parameters: the number of cells per block, the number of pixels per cell, and the number of bins per cell histogram.
 * These blocks typically overlap, meaning that each cell histogram contributes more than once to the final descriptor.
 * To normalize a block its cell histograms \f$h\f$ are grouped together to form a vector \f$v = [h_1, h_2, h_3, ... , h_n]\f$.
 * This vector is normalised using L2-Hys which means performing L2-norm on this vector; clipping the result (by limiting the maximum values of v to be threshold) and renormalizing again. If the threshold is equal to zero then L2-Hys normalization is not performed.
 * \f[L2norm(v) = \frac{v}{\sqrt{\|v\|_2^2 + \epsilon^2}}\f]
 * where \f$ \|v\|_k \f$ be its k-norm for k=1, 2, and \f$ \epsilon \f$ be a small constant.
 * For a specific window its HOG descriptor is then the concatenated vector of the components of the normalized cell histograms from all of the block regions contained in the window.
 * The W1xW2 window starting position is at coordinates 0x0.
 * If the input image has dimensions that are not an integer multiple of W1xW2 blocks with the specified stride, then the last positions that contain only a partial W1xW2 window
 * will be calculated with the remaining part of the W1xW2 window padded with zeroes.
 * The Window W1xW2 must also have a size so that it contains an integer number of cells, otherwise the node is not well-defined.
 * The final output tensor will contain HOG descriptors equal to the number of windows in the input image.
 * The output features tensor has 3 dimensions, given by:\n
 * \f[[ (floor((image_{width}-window_{width})/window_{stride}) + 1),\f]
 * \f[ (floor((image_{height}-window_{height})/window_{stride}) + 1),\f]
 * \f[ floor((window_{width} - block_{width})/block_{stride} + 1) * floor((window_{height} - block_{height})/block_{stride} + 1) *\f]
 * \f[ (((block_{width} * block_{height}) / (cell_{width} * cell_{height})) * num_{bins})] \f]
 * See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>.
 * We recommend the output tensors always be *virtual* objects, with this node connected directly to the classifier.
 * The output tensor will be very large, and using non-virtual tensors will result in a poorly optimized implementation.
 * Merging of this node with a classifier node such as that described in the classifier extension will result in better performance.
 * Notice that this node creation function has more parameters than the corresponding kernel. Numbering of kernel parameters (required if you create this node using the generic interface) is explicitly specified here.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input image of type <tt>\ref VX_DF_IMAGE_U8</tt>. (Kernel parameter #0)
 * \param [in] magnitudes (Optional) The gradient magnitudes per cell of <tt>\ref vx_tensor</tt> of type <tt>\ref VX_TYPE_INT16</tt>. It is the output of <tt>\ref vxHOGCellsNode</tt>.  (Kernel parameter #1)
 * \param [in] bins       The gradient orientation histograms per cell of <tt>\ref vx_tensor</tt> of type <tt>\ref VX_TYPE_INT16</tt>. It is the output of <tt>\ref vxHOGCellsNode</tt>. (Kernel parameter #2)
 * \param [in] params The parameters of type <tt>\ref vx_hog_t</tt>.  (Kernel parameter #3)
 * \param [in] hog_param_size Size of <tt>\ref vx_hog_t</tt> in bytes. Note that this parameter is not counted as one of the kernel parameters.
 * \param [out] features The output HOG features of <tt>\ref vx_tensor</tt> of type <tt>\ref VX_TYPE_INT16</tt>.  (Kernel parameter #4)
 * \return <tt>\ref vx_node</tt>.
 * \retval 0 Node could not be created.
 * \retval * Node handle.
 * \ingroup group_vision_function_hog
 */
VX_API_ENTRY vx_node VX_API_CALL vxHOGFeaturesNode(vx_graph graph, vx_image input, vx_tensor magnitudes, vx_tensor bins, const vx_hog_t *params, vx_size hog_param_size, vx_tensor features);

/*! \brief [Graph] Finds the Probabilistic Hough Lines detected in the input binary image, each line is stored in the output array as a set of points (x1, y1, x2, y2) .
 * \details Some implementations of the algorithm may have a random or non-deterministic element. If the target application is in a safety-critical environment this 
 * should be borne in mind and steps taken in the implementation, the application or both to achieve the level of determinism required by the system design.
 * \param [in] graph graph handle
 * \param [in] input 8 bit, single channel binary source image
 * \param [in] params parameters of the struct <tt>\ref vx_hough_lines_p_t</tt>
 * \param [out] lines_array lines_array contains array of lines, see <tt>\ref vx_line2d_t</tt> The order of lines in implementation dependent
 * \param [out] num_lines [optional] The total number of detected lines in image. Use a VX_TYPE_SIZE scalar
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_hough_lines_p
  */
VX_API_ENTRY vx_node VX_API_CALL vxHoughLinesPNode(vx_graph graph, vx_image input, const vx_hough_lines_p_t *params, vx_array lines_array, vx_scalar num_lines);

/*! \brief [Graph] The function applies bilateral filtering to the input tensor.
* \param [in] graph The reference to the graph.
* \param [in] src The input data a <tt>\ref vx_tensor</tt>. maximum 3 dimension and minimum 2. The tensor is of type <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_INT16</tt>.
* dimensions are [radiometric ,width,height] or [width,height].See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>.
* \param [in] diameter of each pixel neighbourhood that is used during filtering. Values of diameter must be odd. Bigger then 3 and smaller then 10.
* \param [in] sigmaValues Filter sigma in the radiometric space. Supported values are bigger then 0 and smaller or equal 20.
* \param [in] sigmaSpace Filter sigma in the spatial space. Supported values are bigger then 0 and smaller or equal 20.
* \param [out] dst The output data a <tt>\ref vx_tensor</tt>,Of type <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_INT16</tt>. And must be the same type and size of the input.
* \note The border modes
*  <tt>\ref VX_NODE_BORDER</tt> value
*  <tt>\ref VX_BORDER_REPLICATE</tt> and <tt>\ref VX_BORDER_CONSTANT</tt> are supported.
* \return <tt>vx_node</tt>.
* \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>vxGetStatus</tt>
* \ingroup group_vision_function_bilateral_filter
*/
VX_API_ENTRY vx_node VX_API_CALL vxBilateralFilterNode(vx_graph graph, vx_tensor src, vx_int32 diameter, vx_float32 sigmaSpace, vx_float32 sigmaValues, vx_tensor dst);

/*! \brief [Graph] Performs element wise multiplications on element values in the input tensor data with a scale.
 * \param [in] graph The handle to the graph.
 * \param [in] input1 Input tensor data.  Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8,
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt> and <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_position 0.  
 * \param [in] input2 Input tensor data. The dimensions and sizes of input2 match those of input1, unless the vx_tensor of one or more dimensions in input2 is 1.
 * In this case, those dimensions are treated as if this tensor was expanded to match the size of the corresponding dimension of input1,
 * and data was duplicated on all terms in that dimension. After this expansion, the dimensions will be equal.
 * The data type must match the data type of Input1.
 * \param [in] scale A non-negative <tt>\ref VX_TYPE_FLOAT32</tt> multiplied to each product before overflow handling.
 * \param [in] overflow_policy A <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [in] rounding_policy A <tt>\ref vx_round_policy_e</tt> enumeration.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \ingroup group_vision_function_tensor_multiply
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorMultiplyNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_scalar scale, vx_enum overflow_policy,
        vx_enum rounding_policy, vx_tensor output);

/*! \brief [Graph] Performs arithmetic addition on element values in the input tensor data.
 * \param [in] graph The handle to the graph.
 * \param [in] input1 Input tensor data.  Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8,
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt> and <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_position 0.  
 * \param [in] input2 Input tensor data. The dimensions and sizes of input2 match those of input1, unless the vx_tensor of one or more dimensions in input2 is 1.
 * In this case, those dimensions are treated as if this tensor was expanded to match the size of the corresponding dimension of input1,
 * and data was duplicated on all terms in that dimension. After this expansion, the dimensions will be equal. 
 * The data type must match the data type of Input1. 
 * \param [in] policy A <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \ingroup group_vision_function_tensor_add
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorAddNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output);

/*! \brief [Graph] Performs arithmetic subtraction on element values in the input tensor data.
 * \param [in] graph The handle to the graph.
 * \param [in] input1 Input tensor data.  Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8,
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt> and <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_position 0.  
 * \param [in] input2 Input tensor data. The dimensions and sizes of input2 match those of input1, unless the vx_tensor of one or more dimensions in input2 is 1.
 * In this case, those dimensions are treated as if this tensor was expanded to match the size of the corresponding dimension of input1,
 * and data was duplicated on all terms in that dimension. After this expansion, the dimensions will be equal. 
 * The data type must match the data type of Input1. 
 * \param [in] policy A <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \ingroup group_vision_function_tensor_subtract
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorSubtractNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output);

/*! \brief [Graph] Performs LUT on element values in the input tensor data.
 * \param [in] graph The handle to the graph.
 * \param [in] input1 Input tensor data. Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8, 
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt>, with fixed_point_position 0. 
 * \param [in] lut The look-up table to use, of type <tt>\ref vx_lut</tt>.
 * The elements of input1 are treated as unsigned integers to determine an index into the look-up table.
 * The data type of the items in the look-up table must match that of the output tensor.
 * \param [out] output The output tensor data with the same dimensions as the input tensor data.
 * \ingroup group_vision_function_tensor_tablelookup
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorTableLookupNode(vx_graph graph, vx_tensor input1, vx_lut lut, vx_tensor output);

/*! \brief [Graph] Performs transpose on the input tensor.
 * The node transpose the tensor according to a specified 2 indexes in the tensor (0-based indexing)
 * \param [in] graph The handle to the graph.
 * \param [in] input Input tensor data, Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8,
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt> and <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_position 0. 
 * \param [out] output output tensor data,
 * \param [in] dimension1 Dimension index that is transposed with dim 2.
 * \param [in] dimension2 Dimension index that is transposed with dim 1.
 * \ingroup group_vision_function_tensor_transpose
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorTransposeNode(vx_graph graph, vx_tensor input, vx_tensor output, vx_size dimension1, vx_size dimension2);
/*! \brief [Graph] Creates a bit-depth conversion node.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input tensor. Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt> with fixed_point_position 8,
 * and tensor data types <tt>\ref VX_TYPE_UINT8</tt> and <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_position 0.
 * \param [in] policy A <tt>\ref VX_TYPE_ENUM</tt> of the <tt>\ref vx_convert_policy_e</tt> enumeration.
 * \param [in] norm A scalar containing a <tt>\ref VX_TYPE_FLOAT32</tt> of the normalization value.
 * \param [in] offset A scalar containing a <tt>\ref VX_TYPE_FLOAT32</tt> of the offset value subtracted before normalization.
 * \param [out] output The output tensor. Implementations must support input tensor data type <tt>\ref VX_TYPE_INT16</tt>. with fixed_point_position 8. 
 * And <tt>\ref VX_TYPE_UINT8</tt> with fixed_point_position 0.
 * \ingroup group_vision_function_tensor_convert_depth
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorConvertDepthNode(vx_graph graph, vx_tensor input, vx_enum policy, vx_scalar norm, vx_scalar offset, vx_tensor output);

/*! \brief [Graph] Creates a generalized matrix multiplication node.
 * \param [in] graph The reference to the graph.
 * \param [in] input1 The first input 2D tensor of type <tt>\ref  VX_TYPE_INT16</tt> with fixed_point_pos 8, or tensor data types <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_INT8</tt>, with fixed_point_pos 0.
 * \param [in] input2 The second 2D tensor. Must be in the same data type as input1.
 * \param [in] input3 The third 2D tensor. Must be in the same data type as input1. [optional].
 * \param [in] matrix_multiply_params Matrix multiply parameters, see <tt>\ref vx_tensor_matrix_multiply_params_t </tt>.
 * \param [out] output The output 2D tensor. Must be in the same data type as input1. Output dimension must agree the formula in the description.
 * \ingroup group_vision_function_tensor_matrix_multiply
 * \return <tt>\ref vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxTensorMatrixMultiplyNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3,
    const vx_tensor_matrix_multiply_params_t *matrix_multiply_params, vx_tensor output);

/*! \brief Copy data from one object to another.
 * \note An implementation may optimize away the copy when virtual data objects are used.
 * \param [in] graph The reference to the graph.
 * \param [in] input The input data object.
 * \param [out] output The output data object with meta-data identical to the input data object.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation
 * should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_copy
 */
VX_API_ENTRY vx_node VX_API_CALL vxCopyNode(vx_graph graph, vx_reference input, vx_reference output);

/*! \brief Create a batch gemm node, the calcution formula is output = matrix_a * matrix_b + matrix_c.
 * \param [in] graph The reference to the graph.
 * \param [in] matrix_a The first input tensor.
 * \param [in] matrix_b The second input tensor. Must be in the same data type and batch count as first input tensor.
 * \param [in] matrix_c The third input tensor. Must be in the same data type and batch count as first input tensor. [optional]
 * \param [in] trans_a If true, the matrix_a has been transposed before calcution.
 * \param [in] trans_b If true, the matrix_b has been transposed before calcution.
 * \param [in] trans_c If true, the matrix_c has been transposed before calcution. [optional]
 * \param [out] output The output tensor. Output dimension must agree the formula in the description.
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation
 * should be checked using <tt>\ref vxGetStatus</tt>
 * \ingroup group_vision_function_gemm
 */
VX_API_ENTRY vx_node VX_API_CALL vxBatchGemmNode(vx_graph graph,
                                                 vx_tensor matrix_a,
                                                 vx_tensor matrix_b,
                                                 vx_tensor matrix_c,
                                                 vx_scalar trans_a,
                                                 vx_scalar trans_b,
                                                 vx_scalar trans_c,
                                                 vx_tensor output);

#ifdef __cplusplus
}
#endif

#endif
