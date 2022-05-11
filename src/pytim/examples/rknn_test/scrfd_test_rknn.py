import numpy as np
import cv2
from rknn.api import RKNN


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # load rknn
    print('--> Load rknn model')
    ret = rknn.load_rknn("./scrfd.rknn")
    if ret != 0:
        print('Load rknn model failed')
        exit(ret)
    print('done')

    # Set inputs
    input_data = cv2.imread(".examples/rknn_test/scrfd_test.jpg")
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    input_data = input_data.transpose((2,0,1))  # hwc->chw
    input_data = (input_data - 127.5) / 128.0
    input_data = np.round(input_data / 0.0078125 + 128).astype(np.uint8)
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[input_data], inputs_pass_through=[1,])
    print('done')

    print("**********************")
    pred_output = (outputs[2] / 0.0034150767605751753 + 0).astype(np.uint8)
    print(pred_output)
    rknn.release()