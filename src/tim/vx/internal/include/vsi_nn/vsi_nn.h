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
/**
 * @file vsi_nn.h
 */
#ifndef _VSI_NN_INTERFACE_H
#define _VSI_NN_INTERFACE_H

#if defined(_MSC_VER)
#define EXPORT  __declspec(dllexport)
#elif defined(__linux__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

#if !defined(_IN)
#define _IN
#endif
#if !defined(_OUT)
#define _OUT
#endif
#if !defined(_INOUT)
#define _INOUT
#endif
#if !defined(_OPTIONAL)
#define _OPTIONAL
#endif

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

__BEGIN_DECLS


#ifndef TRUE
#define TRUE    (1)
#endif
#ifndef FALSE
#define FALSE   (0)
#endif


/**
 * Return codes.
 */
typedef enum
{
    /**
     * Operation was succesful.
     */
    VSI_NN_ERROR_OK = 0,

    /**
     * Failure caused by vsi_nn api fail.
     */
    VSI_NN_ERROR_API_FAIL = 1,

    /**
     * Failure caused by not enough available memory.
     */
    VSI_NN_ERROR_OUT_OF_MEMORY = 2,

    /**
     * Failure caused by unexpected null argument.
     */
    VSI_NN_ERROR_UNEXPECTED_NULL = 3,

    /**
     * Failure caused by invalid function arguments, invalid model definition,
     * invalid execution definition or invalid data at execution time.
     */
    VSI_NN_ERROR_VALUED_ERROR = 4,

    /**
     * Failure caused by operations that need completed graph.
     */
    VSI_NN_ERROR_UNCOMPLETE_GRAPH = 5,

    /**
     * Failure caused by insearting a keyword argument repeatly.
     */
    VSI_NN_ERROR_KWARGS_REPEAT = 6,
} VSI_NN_error_e;

/**
 * Implicit padding algorithms.
 */
typedef enum
{
    /**
     * Pad with const value which are specific by others parameters.
     */
    VSI_NN_IMPLICIT_PADDING_NONE = 0,

    /**
     * Implicit(VALID) padding.
     * No padding.
     */
    VSI_NN_IMPLICIT_PADDING_VALID = 1,

    /**
     * Implicit(SAME) padding.
     * Padding on both ends are the "same".
     */
    VSI_NN_IMPLICIT_PADDING_SAME = 2,
} VSI_NN_implicit_padding_e;

/**
 * Padding mode.
 */
typedef enum
{
    /**
     * Pad with const value which are specific by others parameters, default 0.
     */
    VSI_NN_PADDING_MODE_CONSTANT = 0,

    /**
     * Reflect padding mode
     */
    VSI_NN_PADDING_MODE_REFLECT = 1,

    /**
     * Symmetric padding mode
     */
    VSI_NN_PADDING_MODE_SYMMETRIC = 2,

    /**
     * Replicate padding mode
     */
    VSI_NN_PADDING_MODE_REPLICATE = 3,
} VSI_NN_padding_mode_e;

/**
 * Rounding methods
 */
typedef enum
{
    /**
     * Floor rounding
     */
    VSI_NN_ROUNDING_FLOOR = 0,
    /**
     * Ceiling rounding
     */
    VSI_NN_ROUNDING_CEIL = 1,
} VSI_NN_rounding_e;

/**
 * LSH Projection supported types.
 */
typedef enum
{
    /**
     * Computed bit vector is considered to be sparse.
     */
    VSI_NN_LSH_PROJECTION_SPARSE = 1,
    /**
     * Computed bit vector is considered to be dense.
     */
    VSI_NN_LSH_PROJECTION_DENSE = 2,
} VSI_NN_lsh_projection_type_e;

/**
 * Supported activation function types.
 */
typedef enum
{
    /** No activation */
    VSI_NN_ACTIVATION_NONE = 0,
    /** ReLU activation */
    VSI_NN_ACTIVATION_RELU = 1,
    /** ReLU1 activation */
    VSI_NN_ACTIVATION_RELU1 = 2,
    /** ReLU6 activation */
    VSI_NN_ACTIVATION_RELU6 = 3,
    /** TanH activation */
    VSI_NN_ACTIVATION_TANH = 4,
    /** Sigmoid activation */
    VSI_NN_ACTIVATION_SIGMOID = 5,
} VSI_NN_activation_e;

/**
 * Tensor types.
 *
 * The type of tensors that can be added to a graph.
 */
typedef enum
{
    /** A tensor of IEEE 754 16 bit floating point values */
    VSI_NN_TENSOR_FLOAT16 = 0,
    /** A tensor of 32 bit floating point values */
    VSI_NN_TENSOR_FLOAT32 = 1,
    /** A tensor of 64 bit floating point values */
    VSI_NN_TENSOR_FLOAT64 = 2,
    /**
     * A tensor of 8 bit boolean values.
     *
     * Values of this operand type are either true or false. A zero value
     * represents false; any other value represents true.
     */
    VSI_NN_TENSOR_BOOL8 = 3,
    /** A tensor of 8 bit integer values */
    VSI_NN_TENSOR_INT8 = 4,
    /** A tensor of 16 bit integer values */
    VSI_NN_TENSOR_INT16 = 5,
    /** A tensor of 32 bit integer values */
    VSI_NN_TENSOR_INT32 = 6,
    /** A tensor of 64 bit integer values */
    VSI_NN_TENSOR_INT64 = 7,
    /** A tensor of 8 bit unsigned integer values */
    VSI_NN_TENSOR_UINT8 = 8,
    /** A tensor of 16 bit unsigned integer values */
    VSI_NN_TENSOR_UINT16 = 9,
    /** A tensor of 32 bit unsigned integer values */
    VSI_NN_TENSOR_UINT32 = 10,
    /** A tensor of 64 bit unsigned integer values */
    VSI_NN_TENSOR_UINT64 = 11,
    /** A tensor of 16 bit truncate floating point values */
    VSI_NN_TENSOR_BFLOAT16 = 12,
} VSI_NN_tensor_type_e;

typedef enum {
    /** Not a quantized tensor */
    VSI_NN_TENSOR_QUANT_NONE = 0,
    /**
     * A tensor of 8 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a number that can be used to convert
     * the 8 bit integer to the real value.
     *
     * fraction_length: a 32 bit signed integer, in range [-128, 127].
     *
     * The formula is:
     *  real_value = integer_value / pow(2, fraction_length).
     */
    VSI_NN_TENSOR_QUANT8_DFP = 1,
    /**
     * A tensor of 16 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a number that can be used to convert
     * the 16 bit integer to the real value.
     *
     * fraction_length: a 32 bit signed integer, in range [-128, 127].
     *
     * The formula is:
     *  real_value = integer_value / pow(2, fraction_length).
     */
    VSI_NN_TENSOR_QUANT16_DFP = 2,
    /**
     * A tensor of 32 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a number that can be used to convert
     * the 16 bit integer to the real value.
     *
     * fraction_length: a 32 bit signed integer, in range [-128, 127].
     *
     * The formula is:
     *  real_value = integer_value / pow(2, fraction_length).
     */
    VSI_NN_TENSOR_QUANT32_DFP = 3,
    /**
     * A tensor of 64 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a number that can be used to convert
     * the 16 bit integer to the real value.
     *
     * fraction_length: a 32 bit signed integer, in range [-128, 127].
     *
     * The formula is:
     *  real_value = integer_value / pow(2, fraction_length).
     */
    VSI_NN_TENSOR_QUANT64_DFP = 4,
    /**
     * A tensor of 8 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a numbers that can be used to convert
     * the 8 bit integer to the real value.
     *
     * scale: a 32 bit floating point value greater than zero.
     *
     * The formula is:
     *  real_value = integer_value * scale.
     */
    VSI_NN_TENSOR_QUANT8_SYMM = 5,
    /**
     * A tensor of 32 bit signed integer values that represent real numbers
     *
     * Attached to this tensor is a numbers that can be used to convert
     * the 8 bit integer to the real value.
     *
     * scale: a 32 bit floating point value greater than zero.
     *
     * The formula is:
     *  real_value = integer_value * scale.
     */
    VSI_NN_TENSOR_QUANT32_SYMM = 6,
    /**
     * A tensor of 8 bit unsigned integer values that represent real numbers
     *
     * Attached to this tensor are two numbers that can be used to convert
     * the 8 bit integer to the real value.
     *
     * scale: a 32 bit floating point value greater than zero.
     * zero_point: a 32 bit signed integer, in range [0, 255].
     *
     * The formula is:
     *  real_value = (integer_value - zero_point) * scale.
     */
    VSI_NN_TENSOR_QUANT8_ASYMM = 7,
    /**
     * A tensor of 8 bit signed integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert
     * the 8 bit integer to the real value.
     *
     * channel_dim: a 32 bit unsigned integer indicating channel dimension.
     * scales: an array of positive 32 bit floating point values.
     * The size of the scales array must be equal to shape[channel_dim].
     *
     * The formula is:
     * realValue[..., C, ...] = integerValue[..., C, ...] * scales[C]
     * where C is an index in the Channel dimension.
     */
    VSI_NN_TENSOR_QUANT8_PERCHANNEL_SYMM = 8,
    /**
     * A tensor of 32 bit signed integers that represent real numbers.
     *
     * Attached to this tensor are two numbers that can be used to convert
     * the 8 bit integer to the real value.
     *
     * channel_dim: a 32 bit unsigned integer indicating channel dimension.
     * scales: an array of positive 32 bit floating point values.
     * The size of the scales array must be equal to shape[channel_dim].
     *
     * The formula is:
     * realValue[..., C, ...] = integerValue[..., C, ...] * scales[C]
     * where C is an index in the Channel dimension.
     */
    VSI_NN_TENSOR_QUANT32_PERCHANNEL_SYMM = 9,
} VSI_NN_tensor_quant_type_e;

/** Parameters for VSI_NN_TENSOR_QUANT8_ASYMM */
typedef struct
{
    float   scale;
    int32_t zero_point;
} VSI_NN_quant_param_asymm;

/** Parameters for VSI_NN_TENSOR_QUANT8_SYMM */
typedef struct
{
    float   scale;
} VSI_NN_quant_param_symm;

/** Parameters for VSI_NN_TENSOR_QUANT8_DFP */
typedef struct
{
    int32_t fraction_length;
} VSI_NN_quant_param_dfp;

/** Parameters for VSI_NN_TENSOR_QUANT8_PERCHANNEL_SYMM */
typedef struct
{
    /** The index of the channel dimension. */
    int32_t channel_dim;

    /**
     * The array of scaling values for each channel.
     * Each value must be greater than zero.
     */
    const float* scales;

    /**
     * The size of the scale array.
     * Should be equal to shape[channel_dim] of the tensor.
     * */
    int32_t scale_count;
} VSI_NN_quant_param_perchannel_symm;

/** Parameters for quantization */
typedef struct
{
    /** Tensor quantize type */
    VSI_NN_tensor_quant_type_e type;
    union
    {
        /** Dynamic fixed point quantization */
        VSI_NN_quant_param_dfp dfp;
        /** Asymmetric affine quantization */
        VSI_NN_quant_param_asymm asymm;
        /** Symmetric affine quantization */
        VSI_NN_quant_param_symm symm;
        /** Perchannel symmetric affine quantization */
        VSI_NN_quant_param_perchannel_symm perchannel_symm;
    } param;
} VSI_NN_tensor_quant_param;

/**
 * NN Runtime context
 */
typedef struct _vsi_nn_context_t VSI_NN_context;

/**
 * VSI_NN_graph is an opaque type that contains a description of the network operations.
 *
 * Create graph by calling VSI_NN_graph_create.
 * A graph is completed by calling VSI_NN_graph_verify.
 * A graph is destroyed by calling VSI_NN_graph_release.
 *
 */
typedef struct _vsi_nn_graph VSI_NN_graph;

/**
 * VSI_NN_tensor is an opaque type that can be used to describe a tensor.
 *
 * Create tensor by calling VSI_NN_tensor_create.
 *
 */
typedef struct _vsi_nn_tensor VSI_NN_tensor;

/**
 * Create context
 *
 * @return Context handle on success or NULL otherwise.
 */
EXPORT VSI_NN_context* VSI_NN_context_create();

/**
 *  Release context
 *
 * @param[in] ctx_ptr The pointer to context to release, and reset point to null.
 */
EXPORT void VSI_NN_context_release
    (
    _IN VSI_NN_context** ctx_ptr
    );

/**
 * Create graph
 * Create a net graph.
 *
 * @param[in] ctx The context used to create graph.
 * @return The graph on success, or NULL otherwise.
 */
EXPORT VSI_NN_graph* VSI_NN_graph_create
    (
    VSI_NN_context* ctx
    );

/**
 * Release graph
 * Release a graph and free its resource.
 *
 * @param[in] graph_ptr The graph to be release.
 */
EXPORT void VSI_NN_graph_release
    (
    _IN VSI_NN_graph** graph_ptr
    );

/**
 * Identify graph inputs and outputs
 * Identify the input and output tensors of a graph. User should call this to
 * specific the inputs and outputs, they are used to exchange data between application
 * level and VSI_NN level.
 *
 * @param[in] graph The graph to be identify.
 * @param[in] input_tensors Input tensors.
 * @param[in] input_tensors_num Number of input tensors.
 * @param[in] output_tensors Output tensors.
 * @param[in] output_tensors_num Number of output tensors.
 * @return VSI_NN_ERROR_OK on success
 */
EXPORT VSI_NN_error_e VSI_NN_graph_identify_input_output
    (
    _IN VSI_NN_graph* graph,
    _IN const VSI_NN_tensor** input_tensors,
    _IN const int32_t input_tensors_num,
    _IN const VSI_NN_tensor** output_tensors,
    _IN const int32_t output_tensors_num
    );

/**
 * To freeze a graph with verifying and compiling.
 *
 * This function may take a long time to compile the graph, and it must only be called
 * once for a given graph.
 *
 * A frozen graph cannot be modified.
 *
 * @param[in] graph The graph to be finished.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_graph_verify
    (
    _IN VSI_NN_graph* graph
    );

/**
 * Compute a frozen graph.
 *
 * @param[in] graph The graph to be executed.
 *
 * @return VSI_NN_ERROR_OK on success. VSI_NN_ERROR_UNCOMPLETE_GRAPH if
 *         the graph is not finished.
 */
EXPORT VSI_NN_error_e VSI_NN_graph_compute
    (
    _IN const VSI_NN_graph* graph
    );

//EXPORT VSI_NN_error_e VSI_NN_GRPAH_profile(_IN const VSI_NN_graph* graph);

/**
 * Add a tensor to a graph.
 *
 * @param[in] graph The graph to be added.
 * @param[in] dtype The data type.
 * @param[in] shape The shape for the tensor.
 * @param[in] ndim  The rank for the tensor.
 * @param[in] memory The memory address to the data, the memory address
 *            must be 64-byte align. If it's set to null, vsi_nn can
 *            optimize the memory allocation and this is default behavior.
 * @param[in] memory_size The size of memory.
 * @param[in] quant_param The quantization parameters for the tensor, set
 *            null if it's not quantized tensor.
 *
 * @return Tensor handle on success, or NULL if get failure.
 */
EXPORT VSI_NN_tensor* VSI_NN_tensor_create
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor_type_e dtype,
    _IN const int32_t* shape,
    _IN int32_t ndim,
    _IN const VSI_NN_tensor_quant_param* quant_param,
    _IN void* memory,
    _IN size_t memory_size,
    _IN int32_t is_constant
    );

/**
 * Add a virtual tensor to a graph.
 *
 * @param[in] graph The graph to be added.
 * @param[in] dtype The data type.
 *
 * @return Tensor handle on success, or NULL if get failure.
 */
EXPORT VSI_NN_tensor* VSI_NN_tensor_create_virtual
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor_type_e dtype,
    _IN const VSI_NN_tensor_quant_param* quant_param
    );

/**
 * Get element size of a tensor.
 *
 * @param[in] tensor Tensor to query element size.
 *
 * @return Element size of the tensor.
 */
EXPORT int32_t VSI_NN_tensor_get_size
    (
    _IN const VSI_NN_tensor* tensor
    );

/**
 * Get bytes of a tensor.
 *
 * @param[in] tensor Tensor to query element size.
 *
 * @return Bytes of the tensor.
 */
EXPORT int32_t VSI_NN_tensor_get_bytes
    (
    _IN const VSI_NN_tensor* tensor
    );

/**
 * Read tensor data.
 *
 * @param[in] tensor Tensor to read.
 * @param[in] memory Memory to fill the data.
 * @param[in] memory_size Element size of the read data,
 *            must be equal to tensor size.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_tensor_read
    (
    _IN VSI_NN_tensor* tensor,
    _IN void* memory,
    _IN size_t memory_size
    );

/**
 * Write data to tensor.
 *
 * @param[in] tensor Tensor to write.
 * @param[in] memory Memory with the data.
 * @param[in] memory_size Element size of the write data,
 *            must be equal to tensor size.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_tensor_write
    (
    _IN VSI_NN_tensor* tensor,
    _IN void* memory,
    _IN size_t memory_size
    );

/**
 * Swap tensors' memories.
 *
 * @param[in] tensor1 Tensor to swap the memory.
 * @param[in] tensor2 Tensor to swap the memory.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_tensor_swap
    (
    _IN VSI_NN_tensor* tensor1,
    _IN VSI_NN_tensor* tensor2
    );

/**
 * Swap tensor memories.
 * User can use this api to get tensor's original memory.
 *
 * @param[in] tensor Tensor to swap the memory.
 * @param[in] new_memory The new memory for the tensor,
 *            if NULL, there is no memory swapped.
 * @param[in] old_memory Pointer for the tensor's original memory.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_tensor_swap_memory
    (
    _IN VSI_NN_tensor* tensor,
    _IN _OPTIONAL void* new_memory,
    _INOUT void** old_memory
    );

/**
 * Flush tensor memory
 * Once a tensor's memory is dirty, user should call this api to sync NPU memory.
 *
 * @param[in] tensor Tensor to flush memory
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_tensor_flush_memory
    (
    _IN const VSI_NN_tensor* tensor
    );

/** Convolutional */
/**
 * Convolution 1D node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] kernel Kernel with a 3D tensor.
 * @param[in] bias Bias with a 1D tensor.
 * @param[in] output Node output tensor.
 * @param[in] stride Convolution stride.
 * @param[in] dilation Convolution dilation rate.
 * @param[in] pad_front Padding front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_end Padding end value.
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] implicit_padding Implicit_padding with value VSI_NN_implicit_padding_e.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_conv_1d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride,
    _IN int32_t dilation,
    _IN int32_t pad_front, _IN int32_t pad_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    );

/**
 * Convolution 2D node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] kernel Kernel with a 4D tensor.
 * @param[in] bias Bias with a 1D tensor.
 * @param[in] output Node output tensor.
 * @param[in] stride_h Convolution stride height.
 * @param[in] stride_w Convolution stride width.
 * @param[in] dilation_h Convolution height dilation rate.
 * @param[in] dilation_w Convolution width dilation rate.
 * @param[in] pad_h_front Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_h_end Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_front Padding width front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_end Padding widht front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] implicit_padding Implicit_padding with value VSI_NN_implicit_padding_e.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    );

/**
 * Depthwise Convolution 2D node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] kernel Kernel with a 4D tensor.
 * @param[in] bias Bias with a 1D tensor.
 * @param[in] output Node output tensor.
 * @param[in] multiplier Depthwise convolution multiplier.
 * @param[in] stride_h Convolution stride height.
 * @param[in] stride_w Convolution stride width.
 * @param[in] dilation_h Convolution height dilation rate.
 * @param[in] dilation_w Convolution width dilation rate.
 * @param[in] pad_h_front Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_h_end Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_front Padding width front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_end Padding widht front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] implicit_padding Implicit_padding with value VSI_NN_implicit_padding_e.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_depthwise_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t multiplier,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    );

/**
 * Grouped Convolution 2D node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] kernel Kernel with a 4D tensor.
 * @param[in] bias Bias with a 1D tensor.
 * @param[in] output Node output tensor.
 * @param[in] group_number Group number for the convolution.
 * @param[in] stride_h Convolution stride height.
 * @param[in] stride_w Convolution stride width.
 * @param[in] dilation_h Convolution height dilation rate.
 * @param[in] dilation_w Convolution width dilation rate.
 * @param[in] pad_h_front Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_h_end Padding height front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_front Padding width front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] pad_w_end Padding widht front value,
 *            this field only effect when implicit
 *            padding is VSI_NN_IMPLICIT_PADDING_NONE.
 * @param[in] implicit_padding Implicit_padding with value VSI_NN_implicit_padding_e.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_grouped_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t group_number,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding
    );

EXPORT VSI_NN_error_e VSI_NN_node_transposed_conv_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t dilation_h, _IN int32_t dilation_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN int32_t output_pad_h, _IN int32_t output_pad_w
    );

/** Pooling */
EXPORT VSI_NN_error_e VSI_NN_node_average_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    );

EXPORT VSI_NN_error_e VSI_NN_node_max_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    );

EXPORT VSI_NN_error_e VSI_NN_node_l2_pool_2d
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t ksize_h, _IN int32_t ksize_w,
    _IN int32_t stride_h, _IN int32_t stride_w,
    _IN int32_t pad_h_front, _IN int32_t pad_h_end,
    _IN int32_t pad_w_front, _IN int32_t pad_w_end,
    _IN VSI_NN_implicit_padding_e implicit_padding,
    _IN VSI_NN_rounding_e size_rounding
    );

EXPORT VSI_NN_error_e VSI_NN_node_unpool_2d();

/** Normalization */
EXPORT VSI_NN_error_e VSI_NN_node_batch_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* mean,
    _IN VSI_NN_tensor* variance,
    _IN VSI_NN_tensor* offset,
    _IN VSI_NN_tensor* scale,
    _IN VSI_NN_tensor* output,
    _IN float variance_epsilon
    );

/**
 * L2 Normalization node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axis Normalize axis.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_l2_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_local_response_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t depth_radius,
    _IN float bias,
    _IN float alpha,
    _IN float beta,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_instance_normalization
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* offset,
    _IN VSI_NN_tensor* scale,
    _IN VSI_NN_tensor* output,
    _IN float variance_epsilon
    );

/** Math */
/**
 * Add node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_add
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Multiply node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_mul
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Divide node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_div
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Subtract node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_sub
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Floor node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_floor
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Square node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_square
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Sqrt node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_sqrt
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Rsqrt node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_rsqrt
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Matmul node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] transpose_input1 Whether to do transpose on input1.
 * @param[in] transpose_input2 Whether to do transpose on input2.
 * @param[in] transpose_output Whether to do transpose on output.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_matmul
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output,
    _IN int transpose_input1,
    _IN int transpose_input2,
    _IN int transpose_output
    );

/**
 * Abs node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_abs
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Pow node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_pow
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Maximum node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_maximum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Minimum node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_minimum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Exp node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_exp
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Reverse node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axes Axes to reverse.
 * @param[in] axes_size Number of axis to reverse.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_reverse
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size
    );

/**
 * Transpose node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] perm Transpose order.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_transpose
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* perm
    );

EXPORT VSI_NN_error_e VSI_NN_node_gather
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* indices,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

/**
 * Neg node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_neg
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Reduce max node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axes Axes to reduce.
 * @param[in] axes_size Number of axis to reduce.
 * @param[in] keep_dim Whether to keep dims on output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_reduce_max
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    );

/**
 * Reduce min node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axes Axes to reduce.
 * @param[in] axes_size Number of axis to reduce.
 * @param[in] keep_dim Whether to keep dims on output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_reduce_min
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    );

/**
 * Reduce sum node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axes Axes to reduce.
 * @param[in] axes_size Number of axis to reduce.
 * @param[in] keep_dim Whether to keep dims on output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_reduce_sum
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    );

/**
 * Reduce mean node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 * @param[in] axes Axes to reduce.
 * @param[in] axes_size Number of axis to reduce.
 * @param[in] keep_dim Whether to keep dims on output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_reduce_mean
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* axes,
    _IN int32_t axes_size,
    _IN int32_t keep_dim
    );

/**
 * Sin node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_sin
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

EXPORT VSI_NN_error_e VSI_NN_node_tile
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* multiples,
    _IN int32_t multiples_size
    );

EXPORT VSI_NN_error_e VSI_NN_node_topk
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_indices,
    _IN int32_t k
    );

/** Logical */
/**
 * Equal node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Greater node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_greater
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Greater equal node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_greater_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Less node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_less
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Less equal node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_less_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Logical and node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_logical_and
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Logical or node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_logical_or
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Logical not node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_logical_not
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Not equal node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_not_equal
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/**
 * Select node.
 * If conditon is true, then output input1 tensor,
 * else output input2 tensor.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] condition Conditon tensor..
 * @param[in] input1 Node input tensor.
 * @param[in] input2 Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_select
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* condition,
    _IN VSI_NN_tensor* input1,
    _IN VSI_NN_tensor* input2,
    _IN VSI_NN_tensor* output
    );

/** Activation */
/**
 * relu node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * ReLU1 node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_relu1
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * ReLU6 node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_relu6
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

EXPORT VSI_NN_error_e VSI_NN_node_tanh
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN float scale_a,
    _IN float scale_b
    );

/**
 * Sigmoid node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_sigmoid
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Hard sigmoid node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_hard_sigmoid
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Mish node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_mish
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

EXPORT VSI_NN_error_e VSI_NN_node_leaky_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN float ratio
    );

EXPORT VSI_NN_error_e VSI_NN_node_prelu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* alpha,
    _IN VSI_NN_tensor* output
    );

/**
 * Soft relu node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_soft_relu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Elu node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_elu
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/** Misc */
EXPORT VSI_NN_error_e VSI_NN_node_pad
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_padding_mode_e mode,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end,
    _IN int32_t pad_value
    );

EXPORT VSI_NN_error_e VSI_NN_node_fully_connected
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* kernel,
    _IN _OPTIONAL VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_concate
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* const inputs[],
    _IN int32_t input_num,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_split
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* const outputs[],
    _IN int32_t output_num,
    _IN const int32_t* slices,
    _IN int32_t slices_size,
    _IN int32_t axis
    );

/**
 * Cast node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_cast
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Quantize node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_quantize
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

/**
 * Dequantize node.
 *
 * @param[in] graph Graph to create the node.
 * @param[in] input Node input tensor.
 * @param[in] output Node output tensor.
 *
 * @return VSI_NN_ERROR_OK on success.
 */
EXPORT VSI_NN_error_e VSI_NN_node_dequantize
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output
    );

EXPORT VSI_NN_error_e VSI_NN_node_space_to_batch
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end
    );

EXPORT VSI_NN_error_e VSI_NN_node_batch_to_space
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* crop_front,
    _IN const int32_t* crop_end
    );

EXPORT VSI_NN_error_e VSI_NN_node_space_to_depth
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* pad_front,
    _IN const int32_t* pad_end
    );

EXPORT VSI_NN_error_e VSI_NN_node_depth_to_space
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* block_size,
    _IN int32_t block_size_num,
    _IN const int32_t* crop_front,
    _IN const int32_t* crop_end
    );

EXPORT VSI_NN_error_e VSI_NN_node_channel_shuffle
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t group_number,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_expand_dims
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_hashtable_lookup
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* lookups,
    _IN VSI_NN_tensor* keys,
    _IN VSI_NN_tensor* values,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_hits
    );

EXPORT VSI_NN_error_e VSI_NN_node_embedding_lookup
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* lookups,
    _IN VSI_NN_tensor* values,
    _IN VSI_NN_tensor* output
     );

EXPORT VSI_NN_error_e VSI_NN_node_lsh_projection
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* hash_func,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* weight,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_lsh_projection_type_e type
    );

EXPORT VSI_NN_error_e VSI_NN_node_slice
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* begin,
    _IN const int32_t* size
    );

EXPORT VSI_NN_error_e VSI_NN_node_strided_slice
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN const int32_t* begin,
    _IN const int32_t* end,
    _IN const int32_t* strides,
    _IN int32_t begin_mask,
    _IN int32_t end_mask,
    _IN int32_t shrink_axis_mask
    );

EXPORT VSI_NN_error_e VSI_NN_node_argmax
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

EXPORT VSI_NN_error_e VSI_NN_node_argmin
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t axis
    );

/** Detection */
EXPORT VSI_NN_error_e VSI_NN_node_roi_pool
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* feature_map,
    _IN VSI_NN_tensor* loc,
    _IN VSI_NN_tensor* batch_index,
    _IN VSI_NN_tensor* output,
    _IN int32_t output_h,
    _IN int32_t output_w,
    _IN float ratio_h,
    _IN float ratio_w
    );

EXPORT VSI_NN_error_e VSI_NN_node_roi_align
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* feature_map,
    _IN VSI_NN_tensor* loc,
    _IN VSI_NN_tensor* batch_index,
    _IN VSI_NN_tensor* output,
    _IN int32_t output_h,
    _IN int32_t output_w,
    _IN float ratio_h,
    _IN float ratio_w,
    _IN int32_t sample_num_h,
    _IN int32_t sample_num_w
    );

/** Image transform */
EXPORT VSI_NN_error_e VSI_NN_node_resize_bilinear
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t scale_h,
    _IN int32_t scale_w
    );

EXPORT VSI_NN_error_e VSI_NN_node_resize_nearest
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* output,
    _IN int32_t scale_h,
    _IN int32_t scale_w
    );

/** RNN */
EXPORT VSI_NN_error_e VSI_NN_node_svdf
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* weights_feature,
    _IN VSI_NN_tensor* weights_time,
    _IN VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* input_state,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_state,
    _IN int32_t rank
    );

//EXPORT VSI_NN_error_e VSI_NN_node_rnn();

EXPORT VSI_NN_error_e VSI_NN_node_rnn_unit
    (
    _IN VSI_NN_graph* graph,
    _IN VSI_NN_tensor* input,
    _IN VSI_NN_tensor* input_state,
    _IN VSI_NN_tensor* weight, _IN VSI_NN_tensor* recrrent_weight,
    _IN VSI_NN_tensor* bias,
    _IN VSI_NN_tensor* output,
    _IN VSI_NN_tensor* output_state,
    _IN VSI_NN_activation_e activation
    );

EXPORT VSI_NN_error_e VSI_NN_node_lstm_unit
    (
    _IN VSI_NN_graph* graph
    );

__END_DECLS
#endif
