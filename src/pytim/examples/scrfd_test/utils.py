import numpy as np
import cv2
import time
import math
import glob
import os
import time

def letterbox(img, new_shape=(480, 320), color=(0, 0, 0), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)

def generate_anchors(base_size, ratios, scales):
    num_ratio = len(ratios)
    num_scale = len(scales)
    anchors = np.zeros((num_ratio * num_scale, 4))
    cx = 0
    cy = 0
    for i in range(num_ratio):
        ar = ratios[i]
        r_w = round(base_size / math.sqrt(ar))
        r_h = round(r_w * ar)
        for j in range(num_scale):
            scale = scales[j]
            rs_w = r_w * scale
            rs_h = r_h * scale
            anchors[i * num_scale + j, 0] = cx - rs_w * 0.5
            anchors[i * num_scale + j, 1] = cy - rs_h * 0.5
            anchors[i * num_scale + j, 2] = cx + rs_w * 0.5
            anchors[i * num_scale + j, 3] = cy + rs_h * 0.5
    return anchors 

def generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects):
    shape = score_blob.shape
    w = shape[1]
    h = shape[0]
    num_anchors = anchors.shape[0]
    for q in range(num_anchors):
        anchor = anchors[q]
        score = score_blob[:, :, q]
        bbox = bbox_blob[:, :, q*4 : q*4 + 4]
        anchor_y = anchor[1]
        anchor_w = anchor[2] - anchor[0]
        anchor_h = anchor[3] - anchor[1]
        for i in range(h):
            anchor_x = anchor[0]
            for j in range(w):
                prob = score[i, j]
                if prob >= prob_threshold:
                    dx = bbox[i,j,0] * feat_stride
                    dy = bbox[i,j,1] * feat_stride
                    dw = bbox[i,j,2] * feat_stride
                    dh = bbox[i,j,3] * feat_stride
                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5
                    x0 = cx - dx
                    y0 = cy - dy
                    x1 = cx + dw
                    y1 = cy + dh
                    obj = {}
                    obj["rect"] = {}
                    obj["rect"]["x0"] = x0
                    obj["rect"]["y0"] = y0
                    obj["rect"]["x1"] = x1
                    obj["rect"]["y1"] = y1
                    obj["rect"]["width"] = x1 - x0 + 1
                    obj["rect"]["height"] = y1 - y0 + 1
                    obj["prob"] = prob
                    obj["landmark"] = []
                    # landmark
                    kps = kps_blob[:, :, q*10 : q*10 + 10]
                    obj["landmark"].append(cx + kps[i, j, 0] * feat_stride)
                    obj["landmark"].append(cy + kps[i, j, 1] * feat_stride)
                    obj["landmark"].append(cx + kps[i, j, 2] * feat_stride)
                    obj["landmark"].append(cy + kps[i, j, 3] * feat_stride)
                    obj["landmark"].append(cx + kps[i, j, 4] * feat_stride)
                    obj["landmark"].append(cy + kps[i, j, 5] * feat_stride)
                    obj["landmark"].append(cx + kps[i, j, 6] * feat_stride)
                    obj["landmark"].append(cy + kps[i, j, 7] * feat_stride)
                    obj["landmark"].append(cx + kps[i, j, 8] * feat_stride)
                    obj["landmark"].append(cy + kps[i, j, 9] * feat_stride)
                    faceobjects.append(obj)
                anchor_x += feat_stride
            anchor_y += feat_stride


def generate_proposals_opt(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects):
    shape = score_blob.shape
    w = shape[1]
    h = shape[0]
    num_anchors = anchors.shape[0]
    for q in range(num_anchors):
        anchor = anchors[q]
        score = score_blob[:, :, q]
        bbox = bbox_blob[:, :, q*4 : q*4 + 4]
        kps = kps_blob[:, :, q*10 : q*10 + 10]
        

        
        
        anchor_x = anchor[0]
        anchor_y = anchor[1]
        anchor_w = anchor[2] - anchor[0]
        anchor_h = anchor[3] - anchor[1]
        cy, cx = np.mgrid[0:h:1, 0:w:1]
        
        cx = cx * feat_stride  + anchor_x
        cy = cy * feat_stride  + anchor_y
        


        index_i, index_j = np.where(score > prob_threshold)
        # index_j, index_i = np.where(score > prob_threshold)
        if len(index_i):

            bbox_new = bbox[index_i, index_j, :] * feat_stride
            kps_new = kps[index_i, index_j, :] * feat_stride
            prob_new = score[index_i, index_j]
            cx_new = cx[index_i, index_j] + anchor_w * 0.5
            cy_new = cy[index_i, index_j] + anchor_h * 0.5
            x0 = cx_new - bbox_new[:, 0]
            y0 = cy_new - bbox_new[:, 1]
            x1 = cx_new + bbox_new[:, 2]
            y1 = cy_new + bbox_new[:, 3]
            p1_x = cx_new + kps_new[:, 0]
            p1_y = cy_new + kps_new[:, 1]
            p2_x = cx_new + kps_new[:, 2]
            p2_y = cy_new + kps_new[:, 3]
            p3_x = cx_new + kps_new[:, 4]
            p3_y = cy_new + kps_new[:, 5]
            p4_x = cx_new + kps_new[:, 6]
            p4_y = cy_new + kps_new[:, 7]
            p5_x = cx_new + kps_new[:, 8]
            p5_y = cy_new + kps_new[:, 9]
            for i in range(len(index_i)):
                obj = {}
                obj["rect"] = {}
                obj["rect"]["x0"] = x0[i]
                obj["rect"]["y0"] = y0[i]
                obj["rect"]["x1"] = x1[i]
                obj["rect"]["y1"] = y1[i]
                obj["rect"]["width"] = x1[i] - x0[i] + 1
                obj["rect"]["height"] = y1[i] - y0[i] + 1                
                obj["prob"] = prob_new[i]
                obj["landmark"] = []
                obj["landmark"].append(p1_x[i])
                obj["landmark"].append(p1_y[i])
                obj["landmark"].append(p2_x[i])
                obj["landmark"].append(p2_y[i])
                obj["landmark"].append(p3_x[i])
                obj["landmark"].append(p3_y[i])
                obj["landmark"].append(p4_x[i])
                obj["landmark"].append(p4_y[i])
                obj["landmark"].append(p5_x[i])
                obj["landmark"].append(p5_y[i])
                faceobjects.append(obj)

def intersection_area(a, b):
    min_x = max(a["rect"]["x0"], b["rect"]["x0"])
    min_y = max(a["rect"]["y0"], b["rect"]["y0"])
    max_x = min(a["rect"]["x1"], b["rect"]["x1"])
    max_y = min(a["rect"]["y1"], b["rect"]["y1"])
    w = max(0, max_x-min_x+1)
    h = max(0, max_y-min_y+1)
    return w * h

def nms_sorted_bboxes(faceobjects, nms_threshold):
    picked = []
    n = len(faceobjects)
    areas = np.zeros((n, ))
    for i in range(n):
        areas[i] = faceobjects[i]["rect"]["width"] * faceobjects[i]["rect"]["height"]
    for i in range(n):
        a = faceobjects[i]
        keep = 1
        for j in range(len(picked)):
            b = faceobjects[picked[j]]
            inter_area = intersection_area(a, b)
            union_area = areas[i] + areas[picked[j]] - inter_area
            if inter_area / union_area > nms_threshold:
                keep = 0
        if keep:
            picked.append(i)
    return picked

def sort_bboxes(face_objs):
    sorted_face_objs = []
    while len(face_objs):
        n = len(face_objs)
        max_prob_index = -1
        max_prob = 0
        for i in range(n):
            if max_prob < face_objs[i]["prob"]:
                max_prob = face_objs[i]["prob"]
                max_prob_index = i
        if max_prob_index != -1:
            sorted_face_objs.append(face_objs[max_prob_index])
            del face_objs[max_prob_index]

    return sorted_face_objs


def decode(ratio, pad, src_img, input_img, net_out_tensors, prob_threshold, nms_threshold):

    net_h, net_w, net_c = input_img.shape
    ratios = []
    ratios.append(1.0)
    scales = []
    scales.append(1.0)
    scales.append(2.0)
    faceproposals = []
    # stride 8
    faceobjects = []
    base_size = 16
    feat_stride = 8
    h = int(net_h / feat_stride)
    w = int(net_w / feat_stride)
    c = len(scales)
    score_blob_stride8 = net_out_tensors[0].reshape((h, w, c))
    bbox_blob_stride8 = net_out_tensors[3].reshape((h, w, c*4))
    kps_blob_stride8 = net_out_tensors[6].reshape((h, w, c*10))
    anchors = generate_anchors(base_size, ratios, scales)
    generate_proposals_opt(anchors, feat_stride, score_blob_stride8, bbox_blob_stride8, kps_blob_stride8, prob_threshold, faceobjects)
    faceproposals.extend(faceobjects)
    # stride 16
    faceobjects = []
    base_size = 64
    feat_stride = 16
    h = int(net_h / feat_stride)
    w = int(net_w / feat_stride)
    c = len(scales)
    score_blob_stride16 = net_out_tensors[1].reshape((h, w, c))
    bbox_blob_stride16 = net_out_tensors[4].reshape((h, w, c*4))
    kps_blob_stride16 = net_out_tensors[7].reshape((h, w, c*10))
    anchors = generate_anchors(base_size, ratios, scales)
    generate_proposals_opt(anchors, feat_stride, score_blob_stride16, bbox_blob_stride16, kps_blob_stride16, prob_threshold, faceobjects)
    faceproposals.extend(faceobjects)
    # stride 32
    faceobjects = []
    base_size = 256
    feat_stride = 32
    h = int(net_h / feat_stride)
    w = int(net_w / feat_stride)
    c = len(scales)    
    score_blob_stride32 = net_out_tensors[2].reshape((h, w, c))
    bbox_blob_stride32 = net_out_tensors[5].reshape((h, w, c*4))
    kps_blob_stride32 = net_out_tensors[8].reshape((h, w, c*10))
    anchors = generate_anchors(base_size, ratios, scales)
    generate_proposals_opt(anchors, feat_stride, score_blob_stride32, bbox_blob_stride32, kps_blob_stride32, prob_threshold, faceobjects)  
    faceproposals.extend(faceobjects)


    faceproposals = sort_bboxes(faceproposals)
    picked = nms_sorted_bboxes(faceproposals, nms_threshold)

    wpad = pad[0]
    hpad = pad[1]
    height = src_img.shape[0]
    width = src_img.shape[1]
    scale_face_objs = []
    scale = ratio[0]
    for i in range(len(picked)):
        face_obj = faceproposals[picked[i]]
        x0 = (face_obj["rect"]["x0"] - wpad) / scale
        y0 = (face_obj["rect"]["y0"] - hpad) / scale
        x1 = (face_obj["rect"]["x1"] - wpad) / scale
        y1 = (face_obj["rect"]["y1"] - hpad) / scale
        x0 = max(min(x0, width - 1), 0)
        y0 = max(min(y0, height - 1), 0)
        x1 = max(min(x1, width - 1), 0)
        y1 = max(min(y1, height - 1), 0)
        face_obj["rect"]["x0"] = x0
        face_obj["rect"]["y0"] = y0
        face_obj["rect"]["x1"] = x1
        face_obj["rect"]["y1"] = y1        
        face_obj["rect"]["width"] = x1 - x0
        face_obj["rect"]["height"] = y1 - y0
        x0 = (face_obj["landmark"][0] - wpad) / scale
        y0 = (face_obj["landmark"][1] - hpad) / scale
        x1 = (face_obj["landmark"][2] - wpad) / scale
        y1 = (face_obj["landmark"][3] - hpad) / scale
        x2 = (face_obj["landmark"][4] - wpad) / scale
        y2 = (face_obj["landmark"][5] - hpad) / scale
        x3 = (face_obj["landmark"][6] - wpad) / scale
        y3 = (face_obj["landmark"][7] - hpad) / scale
        x4 = (face_obj["landmark"][8] - wpad) / scale
        y4 = (face_obj["landmark"][9] - hpad) / scale
        face_obj["landmark"][0] = max(min(x0, width - 1), 0)
        face_obj["landmark"][1] = max(min(y0, height - 1), 0)
        face_obj["landmark"][2] = max(min(x1, width - 1), 0)
        face_obj["landmark"][3] = max(min(y1, height - 1), 0)
        face_obj["landmark"][4] = max(min(x2, width - 1), 0)
        face_obj["landmark"][5] = max(min(y2, height - 1), 0)
        face_obj["landmark"][6] = max(min(x3, width - 1), 0)
        face_obj["landmark"][7] = max(min(y3, height - 1), 0)
        face_obj["landmark"][8] = max(min(x4, width - 1), 0)
        face_obj["landmark"][9] = max(min(y4, height - 1), 0)
        scale_face_objs.append(face_obj)


    return scale_face_objs


def save_detect_faces(detect_faces, save_dir, file_name):
    src_img = cv2.imread(file_name)
    for i in range(len(detect_faces)):
        face = detect_faces[i]
        x0 = int(face["rect"]["x0"])
        y0 = int(face["rect"]["y0"])
        x1 = int(face["rect"]["x1"])
        y1 = int(face["rect"]["y1"])
        pt1 = (x0,y0)
        pt2 = (x1,y1)
        cv2.rectangle(src_img, pt1, pt2, (0, 255, 0), 2)
        point1 = (int(face["landmark"][0]), int(face["landmark"][1]))
        point2 = (int(face["landmark"][2]), int(face["landmark"][3]))
        point3 = (int(face["landmark"][4]), int(face["landmark"][5]))
        point4 = (int(face["landmark"][6]), int(face["landmark"][7]))
        point5 = (int(face["landmark"][8]), int(face["landmark"][9]))
        points_list = [point1, point2, point3, point4, point5]
        point_size = 2
        point_color = (0, 0, 255)
        thickness = 2
        for point in points_list:
            cv2.circle(src_img, point, point_size, point_color, thickness)

    dst_file = os.path.join(save_dir, os.path.basename(file_name))
    cv2.imwrite(dst_file, src_img)
