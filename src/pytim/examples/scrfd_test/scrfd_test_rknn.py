import numpy as np
import cv2
from rknn.api import RKNN
from .utils import letterbox, decode

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # load rknn
    print('--> Load rknn model')
    ret = rknn.load_rknn("./examples/scrfd_test/scrfd.rknn")
    if ret != 0:
        print('Load rknn model failed')
        exit(ret)
    print('done')

    # Set inputs
    src_img = cv2.imread("./examples/scrfd_test/scrfd_test.jpg")
    img,ratio,pad = letterbox(src_img)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')

    # post process
    print("**********************")
    rknn_detect_faces = decode(ratio, pad, src_img, img, outputs, 0.4, 0.45)
    print("detect {} faces".format(len(rknn_detect_faces)))
    print("face location:{}".format(rknn_detect_faces))
    rknn.release()