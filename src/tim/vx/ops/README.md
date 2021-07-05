INFO: Actual implementations may differ from reference link in terms of dimensions and parameters supported

TIM-VX API |Internal Op |Status | Reference
:------    |:----- |:------|:------
Add|ADD|Mapped|[tf.math.add](https://tensorflow.google.cn/api_docs/python/tf/math/add)
Multiply|MULTIPLY|Mapped|[tf.math.multiply](https://tensorflow.google.cn/api_docs/python/tf/math/multiply)
Conv2d|CONV2D|Mapped|[tf.nn.conv2d](https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d) [tf.nn.atros_conv2d](https://tensorflow.google.cn/api_docs/python/tf/nn/atrous_conv2d) [tf.nn.depthwise_conv2d](https://tensorflow.google.cn/api_docs/python/tf/nn/depthwise_conv2d)
Softmax|SOFTMAX|Mapped|[tf.nn.softmax](https://tensorflow.google.cn/api_docs/python/tf/nn/softmax)
Pool2d|POOL|Mapped|[tf.nn.pool](https://tensorflow.google.cn/api_docs/python/tf/nn/pool)
LeakyRelu|LEAKY_RELU|Mapped|[tf.nn.leaky_relu](https://tensorflow.google.cn/api_docs/python/tf/nn/leaky_relu)
Concat|CONCAT|Mapped|[tf.concat](https://tensorflow.google.cn/api_docs/python/tf/concat)
Split|SPLIT|Mapped|[tf.split](https://tensorflow.google.cn/api_docs/python/tf/split)
BatchNorm|BATCH_NORM|Mapped|[tf.nn.batch_normalization](https://tensorflow.google.cn/api_docs/python/tf/nn/batch_normalization)
DeConv2d|DECONVOLUTION|Mapped|[tf.nn.conv2d_transpose](https://tensorflow.google.cn/api_docs/python/tf/nn/conv2d_transpose)
Reshape|RESHAPE|Mapped|[tf.reshape](https://tensorflow.google.cn/api_docs/python/tf/reshape)
Transpose|PERMUTE|Mapped|[tf.transpose](https://tensorflow.google.cn/api_docs/python/tf/transpose)
Prelu|PRELU|Mapped|[tf.keras.layers.PReLU](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/PReLU)
MaxUnpool2d|UPSAMPLE|Mapped|[tfa.layers.MaxUnpooling2D](https://tensorflow.google.cn/addons/api_docs/python/tfa/layers/MaxUnpooling2D)
Relu|RELU|Mapped|[tf.nn.relu](https://tensorflow.google.cn/api_docs/python/tf/nn/relu)
Reorg|REORG|Mapped|[darknet.reorg](https://github.com/pjreddie/darknet/blob/master/src/reorg_layer.c)
L2Normalization|L2_NORMALIZE|Mapped|[tf.math.l2_normalize](https://tensorflow.google.cn/api_docs/python/tf/math/l2_normalize)
FullyConnected|FCL2|Mapped|[tf.keras.layers.Dense](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Dense)
MaxpoolWithArgmax|POOLWITHARGMAX|Mapped|[tf.nn.max_pool_with_argmax](https://tensorflow.google.cn/api_docs/python/tf/nn/max_pool_with_argmax)
ArgMax|ARGMAX|Mapped|[tf.math.argmax](https://tensorflow.google.cn/api_docs/python/tf/math/argmax)
Maximum|MAXIMUM|Mapped|[tf.math.maximum](https://tensorflow.google.cn/api_docs/python/tf/math/maximum)
Sub|SUBTRACT|Mapped|[tf.math.subtract](https://tensorflow.google.cn/api_docs/python/tf/math/subtract)
Relu6|RELU6|Mapped|[tf.nn.relu6](https://tensorflow.google.cn/api_docs/python/tf/nn/relu6)
Sigmoid|SIGMOID|Mapped|[tf.math.sigmoid](https://tensorflow.google.cn/api_docs/python/tf/math/sigmoid)
Tanh|TANH|Mapped|[tf.math.tanh](https://tensorflow.google.cn/api_docs/python/tf/math/tanh)
Sqrt|SQRT|Mapped|[tf.math.sqrt](https://tensorflow.google.cn/api_docs/python/tf/math/sqrt)
Rsqrt|RSQRT|Mapped|[tf.math.rsqrt](https://tensorflow.google.cn/api_docs/python/tf/math/rsqrt)
SoftRelu|SOFTRELU|Mapped|[tf.math.softplus](https://tensorflow.google.cn/api_docs/python/tf/math/softplus)
Div|DIVIDE|Mapped|[tf.math.divide](https://tensorflow.google.cn/api_docs/python/tf/math/divide)
Dropout|DROPOUT|Mapped|f(x) = x\*ratio
Resize|RESIZE|Mapped|[tf.image.resize](https://tensorflow.google.cn/api_docs/python/tf/image/resize)
Reverse|REVERSE|Mapped|[tf.reverse](https://tensorflow.google.cn/api_docs/python/tf/reverse)
DepthToSpace|DEPTH2SPACE|Mapped|[tf.nn.depth_to_space](https://tensorflow.google.cn/api_docs/python/tf/nn/depth_to_space)
SpaceToDepth|SPACE2DEPTH|Mapped|[tf.nn.space_to_depth](https://tensorflow.google.cn/api_docs/python/tf/nn/space_to_depth)
DataConvert|DATACONVERT|Mapped|Data Format Conversion
Slice|SLICE|Mapped|[tf.slice](https://tensorflow.google.cn/api_docs/python/tf/slice)
Elu|ELU|Mapped|[tf.nn.elu](https://tensorflow.google.cn/api_docs/python/tf/nn/elu)
Batch2Space|BATCH2SPACE|Mapped|[tf.batch_to_space](https://tensorflow.google.cn/api_docs/python/tf/batch_to_space)
Space2Batch|SPACE2BATCH|Mapped|[tf.space_to_batch](https://tensorflow.google.cn/api_docs/python/tf/space_to_batch)
Pad|PAD|Mapped|[tf.pad](https://tensorflow.google.cn/api_docs/python/tf/pad)
Matmul|MATRIXMUL|Mapped|[tf.linalg.matmul](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul)
LayerNormalization|LAYER_NORM|Mapped|[tf.keras.layers.LayerNormalization](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/LayerNormalization)
ReduceMin|REDUCE_MIN|Mapped|[tf.math.reduce_min](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_min)
ReduceMax|REDUCE_MAX|Mapped|[tf.math.reduce_max](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_max)
ReduceAny|REDUCE_ANY|Mapped|[tf.math.reduce_any](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_any)
ReduceProd|REDUCE_PROD|Mapped|[tf.math.reduce_prod](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_prod)
ReduceMean|REDUCE_MEAN|Mapped|[tf.math.reduce_mean](https://tensorflow.google.cn/api_docs/python/tf/math/reduce_mean)
InstanceNormalization|INSTANCE_NORM|Mapped|[tfa.layers.InstanceNormalization](https://tensorflow.google.cn/addons/api_docs/python/tfa/layers/InstanceNormalization)
StridedSlice|STRIDED_SLICE|Mapped|[tf.strided_slice](https://tensorflow.google.cn/api_docs/python/tf/strided_slice)
Abs|ABS|Mapped|[tf.math.abs](https://tensorflow.google.cn/api_docs/python/tf/math/abs)
Conv1d|CONV1D|Mapped|[tf.nn.conv1d](https://tensorflow.google.cn/api_docs/python/tf/nn/conv1d)
NBG|NBG|Mapped|Network Binary Graph
LocalResponseNormalization|LRN2|Mapped|[tf.nn.local_response_normalization](https://tensorflow.google.cn/api_docs/python/tf/nn/local_response_normalization)
Greater|RELATIONAL_OPS_GREATER|Mapped|[tf.math.greater](https://tensorflow.google.cn/api_docs/python/tf/math/greater)
GreaterOrEqual|RELATIONAL_OPS_GREATER_EQUAL|Mapped|[tf.math.greater_equal](https://tensorflow.google.cn/api_docs/python/tf/math/greater_equal)
Less|RELATIONAL_OPS_LESS|Mapped|[tf.math.less](https://tensorflow.google.cn/api_docs/python/tf/math/less)
LessOrEqual|RELATIONAL_OPS_LESS_EQUAL|Mapped|[tf.math.less_equal](https://tensorflow.google.cn/api_docs/python/tf/math/less_equal)
Equal|RELATIONAL_OPS_EQUAL|Mapped|[tf.math.equal](https://tensorflow.google.cn/api_docs/python/tf/math/equal)
NotEqual|RELATIONAL_OPS_NOT_EQUAL|Mapped|[tf.math.not_equal](https://tensorflow.google.cn/api_docs/python/tf/math/not_equal)
Pow|POW|Mapped|[tf.math.pow](https://tensorflow.google.cn/api_docs/python/tf/math/pow)
FloorDiv|FLOORDIV|Mapped|[tf.math.floordiv](https://tensorflow.google.cn/api_docs/python/tf/math/floordiv)
Minimum|MINIMUM|Mapped|[tf.math.minimum](https://tensorflow.google.cn/api_docs/python/tf/math/minimum)
And|LOGICAL_OPS|Mapped|[tf.math.logical_and](https://tensorflow.google.cn/api_docs/python/tf/math/logical_and)
Or|LOGICAL_OPS|Mapped|[tf.math.logical_or](https://tensorflow.google.cn/api_docs/python/tf/math/logical_or)
Select|SELECT|Mapped|[tf.where](https://tensorflow.google.cn/api_docs/python/tf/where)
Relu1|RELU1|Mapped|[tf.keras.layers.ReLU(max_value=1.0)](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/ReLU)
Stack|STACK|Mapped|[tf.stack](https://tensorflow.google.cn/api_docs/python/tf/stack)
Floor|FLOOR|Mapped|[tf.math.floor](https://tensorflow.google.cn/api_docs/python/tf/math/floor)
Square|SQUARE|Mapped|[tf.math.square](https://tensorflow.google.cn/api_docs/python/tf/math/square)
Neg|NEG|Mapped|[tf.math.negative](https://tensorflow.google.cn/api_docs/python/tf/math/negative)
Exp|EXP|Mapped|[tf.math.exp](https://tensorflow.google.cn/api_docs/python/tf/math/exp)
Clip|CLIP|Mapped|[tf.clip_by_value](https://tensorflow.google.cn/api_docs/python/tf/clip_by_value)
AddN|ADDN|Mapped|[tf.math.add_n](https://tensorflow.google.cn/api_docs/python/tf/math/add_n)
Gather|GATHER|Mapped|[tf.gather](https://tensorflow.google.cn/api_docs/python/tf/gather)
LogicalNot|LOGICAL_NOT|Mapped|[tf.math.logical_not](https://tensorflow.google.cn/api_docs/python/tf/math/logical_not)
Sin|SIN|Mapped|[tf.math.sin](https://tensorflow.google.cn/api_docs/python/tf/math/sin)
Log|LOG|Mapped|[tf.math.log](https://tensorflow.google.cn/api_docs/python/tf/math/log)
ArgMin|ARGMIN|Mapped|[tf.math.argmin](https://tensorflow.google.cn/api_docs/python/tf/math/argmin)
LogSoftmax|LOG_SOFTMAX|Mapped|[tf.nn.log_softmax](https://tensorflow.google.cn/api_docs/python/tf/nn/log_softmax)
HardSwish|SWISH|Mapped|[tf.keras.activations.swish](https://tensorflow.google.cn/api_docs/python/tf/keras/activations/swish)
GatherNd|GATHER_ND|Mapped|[tf.gather_nd](https://tensorflow.google.cn/api_docs/python/tf/gather_nd)
Cast|CAST|Mapped|[tf.cast](https://tensorflow.google.cn/api_docs/python/tf/cast)
Moments|MOMENTS|Mapped|[tf.moments](https://tensorflow.google.cn/api_docs/python/tf/nn/moments)
Squeeze|SQUEEZE|Mapped|[tf.squeeze](https://tensorflow.google.cn/api_docs/python/tf/squeeze)
HardSigmoid|HARD_SIGMOID|Mapped|[tf.keras.activations.hard_sigmoid](https://tensorflow.google.cn/api_docs/python/tf/keras/activations/hard_sigmoid)
Mish|MISH|Mapped|[tfa.activations.mish](https://tensorflow.google.cn/addons/api_docs/python/tfa/activations/mish)
DeConv1d|DECONVOLUTION1D|Mapped|[tf.nn.conv1d_transpose](https://tensorflow.google.cn/api_docs/python/tf/nn/conv1d_transpose)
Resize1d|RESIZE_1D|Mapped|[Onnx.resize 1D image](https://github.com/onnx/onnx/blob/master/docs/Operators.md#resize)
Linear|LINEAR|Mapped|[tf.keras.activations.linear](https://www.tensorflow.org/api_docs/python/tf/keras/activations/linear)
ScatterND|SCATTER_ND|Mapped|[tf.scatter_nd](https://tensorflow.google.cn/api_docs/python/tf/scatter_nd)
||MOMENTS|Planned 21Q2|[tf.moments](https://tensorflow.google.cn/api_docs/python/tf/nn/moments)
||MATRIXMUL|Planned 21Q2|[tf.experimental.numpy.matmul](https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/matmul)
Unstack|UNSTACK|Mapped|[tf.unstack](https://tensorflow.google.cn/api_docs/python/tf/unstack)
Tile|TILE|Mapped|[tf.tile](https://tensorflow.google.cn/api_docs/python/tf/tile)
GroupedConv2d|GROUPED_CONV2D|Mapped|[ANEURALNETWORKS_GROUPED_CONV_2D](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a847acf8d9f3d2343328c3dbe6d447c50)
SpatialTransformer|SPATIAL_TRANSFORMER|Mapped|[SpatialTransformer](https://github.com/daerduoCarey/SpatialTransformerLayer)
||PROPOSAL|Planned 21Q3|[Faster-RCNN Proposal Layer](https://github.com/intel/caffe/blob/master/examples/faster-rcnn/lib/rpn/proposal_layer.py)
||ROI_POOL|Planned 21Q3|[ANEURALNETWORKS_ROI_POOLING](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a6736198af337b2efbdb0b6b64dee7fe4)
||ROI_ALIGN|Planned 21Q3|[ANEURALNETWORKS_ROI_ALIGN](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a2848b39dd4bfba78f2438fda0d9397a4)
||SHUFFLECHANNEL|Planned 21Q3|[ANEURALNETWORKS_CHANNEL_SHUFFLE](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a5b993c1211c4b1bc52fb595a3025251d)
||SIGNAL_FRAME|Planned 21Q3|[tf.signal.frame](https://tensorflow.google.cn/api_docs/python/tf/signal/frame)
||TOPK|Planned 21Q3|[tf.math.top_k](https://tensorflow.google.cn/api_docs/python/tf/math/top_k)
|GRUCell|GRUCELL_OVXLIB|Planned 21Q3|[tf.keras.layers.GRUCell](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRUCell?hl=en)
|UnidirectionalSequenceGRU|GRU_OVXLIB|Planned 21Q4|[tf.keras.layers.GRU](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/GRUCell?hl=en)
|UnidirectionalSequenceRNN|UNIDIRECTIONAL_SEQUENCE_RNN|Planned 21Q4|[ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_RNN](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0ae11aa1d461d2abaa117f6ee2cb503dd8)
|BidirectionalSequenceRNN|BIDIRECTIONAL_SEQUENCE_RNN|Planned 21Q4|[ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_RNN](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a487fc5ae247de828f13e62b99f259f3c)
|RNNCell|RNNCELL_OVXLIB|Planned 21Q3|[ANEURALNETWORKS_RNN](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0acd2684ac9c73bb29767b534e78a332e8)
|BidirectionalSequenceLSTM|BIDIRECTIONAL_SEQUENCE_LSTM|Planned 21Q4|[ANEURALNETWORKS_BIDIRECTIONAL_SEQUENCE_LSTM](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a492a71cb7aa50b9a1a834a3cb269d778)
|UnidirectionalSequenceLSTM|LSTM_OVXLIB|Planned 21Q4|[ANEURALNETWORKS_UNIDIRECTIONAL_SEQUENCE_LSTM](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0aaf30e491ad0b1fc7602cbde695b2c859)
|LSTMCell|LSTMUNIT_OVXLIB|Planned 21Q3|[ANEURALNETWORKS_LSTM](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0ad0377e8c305e596fb7f64ff896671fc5)
||PRE_PROCESS|Planned 21Q4|Image Preprocessing (YUV2RGB, Input Normalization, Resizing, etc)
||HASHTABLE_LOOKUP|Planned 21Q4|[ANEURALNETWORKS_HASHTABLE_LOOKUP](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0aca92716c8c73c1f0fa7f0757916fee26)
||EMBEDDING_LOOKUP|Planned 21Q4|[ANEURALNETWORKS_EMBEDDING_LOOKUP](developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a8d2ada77adb74357fc0770405bca0e3)
||LSH_PROJECTION|Planned 21Q4|[ANEURALNETWORKS_LSH_PROJECTION](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a800cdcec5d7ba776789cb2d1ef669965)
||SVDF|Planned 21Q4|[ANEURALNETWORKS_SVDF](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a7096de21038c1ce49d354a00cba7b552)
||HEATMAP_MAX_KEYPOINT|Planned 21Q4|[ANEURALNETWORKS_HEATMAP_MAX_KEYPOINT](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a5ffccf92d127766a741225ff7ad6f743)
||AXIS_ALIGNED_BBOX_TRANSFORM|Planned 21Q4|[ANEURALNETWORKS_AXIS_ALIGNED_BBOX_TRANSFORM](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0afd7603dd54060e6a52f5861674448528)
||BOX_WITH_NMS_LIMIT|Planned 21Q4|[ANEURALNETWORKS_BOX_WITH_NMX_LIMIT](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a2d81e878c19e15700dad111ba6c0be89)
||GENERATE_PROPOSALS|Planned 21Q4|[ANEURALNETWORKS_GENERATE_PROPOSALS](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a72484020f2c41c814de0a7bf93dbbfd4)
||DETECTION_POSTPROCESS|Planned 21Q4|[ANEURALNETWORKS_DETECTION_POSTPROCESSING](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0abd6365933837275bb1f5cde1fd9b8234)
||RANDOM_MULTINOMIAL|Planned 21Q4|[ANEURALNETWORKS_RANDOM_MULTINOMIAL](https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a6cb5032c09d3c4b542d18495c247b5b4)
||CONV_RELU|Deprecated
||CONV_RELU_POOL|Deprecated
||FCL|Deprecated
||FCL_RELU|Deprecated
||LRN|Deprecated
||SCALE|Deprecated
||DEPTHWISE_CONV1D|Deprecated
||L2NORMALIZESCALE|Deprecated
||INTERP|Deprecated
||NOOP|Deprecated
||TENSORSTACKCONCAT|Deprecated|
||VARIABLE|InternalOnly|
||RELUN|Deprecated|
||CROP|Deprecated
||TENSOR_ADD_MEAN_STDDEV_NORM|InternalOnly
||RNN|Deprecated|
||LSTMUNIT_ACTIVATION|InternalOnly|
||LSTM|Deprecated|
||LSTMUNIT|Deprecated|
||QUANTIZED_16BIT_LSTM|InternalOnly
||RELU_KERAS|Deprecated|
||PRE_PROCESS_GRAY|InternalOnly
||PRE_PROCESS_YUV444|InternalOnly
||PRE_PROCESS_NV12|InternalOnly
||PRE_PROCESS_YUV420|InternalOnly
||PRE_PROCESS_BGRA|InternalOnly
||PRE_PROCESS_TENSOR|InternalOnly
||IMAGEPROCESS|Deprecated
||POST_PROCESS|InternalOnly
||EXTRA_ENDING|InternalOnly
||SYNC_HOST|InternalOnly
||BATCHNORM_SINGLE|InternalOnly|
||CONCATSHIFT|InternalOnly
||A_TIMES_B_PLUS_C|Deprecated|[tf.add(tf.mul(A, B), C)](https://github.com/hujie-frank/SENet/blob/master/include/caffe/layers/axpy_layer.hpp)
