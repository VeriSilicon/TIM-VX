import numpy as np
import cv2
from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()
    
    # load rknn
    print('--> Load rknn model')
    ret = rknn.load_rknn("./examples/lenet_test/lenet.rknn")
    if ret != 0:
        print('Load rknn model failed')
        exit(ret)
    print('done')

    # Set inputs
    lenet_input_data = np.load("./examples/lenet_test/lenet_input.npy").reshape((1, 1, 28, 28))

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[lenet_input_data])
    print('done')

    # get output 
    outputs[0] = (outputs[0] / 28.09471893310547 + 144).astype(np.uint8)
    print(outputs)

    # release rknn
    rknn.release()