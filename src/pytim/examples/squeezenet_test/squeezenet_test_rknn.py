import numpy as np
import cv2
from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # load rknn
    print('--> Load rknn model')
    ret = rknn.load_rknn("./examples/squeezenet_test/squeezenet.rknn")
    if ret != 0:
        print('Load rknn model failed')
        exit(ret)
    print('done')

    # Set inputs
    input_data = cv2.imread("./examples/squeezenet_test/squeezenet_test.jpg")
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_data])
    print('done')

    print("**********************")
    pred_output = outputs[0].reshape((1000,)).astype(np.float32)
    top_k=5
    top_k_idx=pred_output.argsort()[::-1][0:top_k]    
    print(pred_output[top_k_idx])
    rknn.release()