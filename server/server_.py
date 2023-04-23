#pytorch
from concurrent.futures import thread
from sqlalchemy import null
import torch
from torchvision import transforms
import time
import datetime as dt
from threading import Thread

#other lib
from os import remove
import time

#socket
import os
import socket
import cv2
import numpy as np
import sys
from os.path import exists


sys.path.insert(0, "yolov5_face")
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
model = attempt_load("yolov5_face/yolov5n-0.5.pt", map_location=device)

# Get model recognition
from insightface.insight_face import iresnet100
weight = torch.load("insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

# Resize image
def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 128
    conf_thres = 0.4
    iou_thres = 0.5
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]

    return images_name, images_emb

def make_dir_list(path) :
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    #print(files_dir)
    return files_dir

def recognition(face_image):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))
    
    # Read features
    images_names, images_embs = read_features()   

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    
    name = images_names[id_min]
    isThread = True
    #path = './dataset/face-datasets'
    #file = make_dir_list(path)
    #for i in range(len(file)):
        #if file[i] in name :
            #print(name)
    # print("Recognization Successful")

# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 서버의 아이피와 포트번호 지정
HOST = ''
PORT = 13330
s.bind((HOST, PORT))
print('Socket bind complete')
# 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
s.listen(10)
print('Socket now listening')

# 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
conn, addr = s.accept()

isThread = True
score = 0
name = null

def server():
    # recognize
    global isThread, score, name, now_time, recognition_res

    start = time.time_ns()
    frame_count = 0
    fps = -1
    unknown_cnt = 0
    recognition_res = {}
    start_time = time.time()  # 프로그램이 시작한 시간 저장

    while True:
        # client에서 받은 stringData의 크기 (==(str(len(stringData))).encode().ljust(16))
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        data = np.frombuffer(stringData, dtype='uint8')

        # data를 디코딩한다.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Get faces
        bboxs, landmarks = get_face(frame)
        h, w, c = frame.shape
        
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        
        # Get boxs
        for i in range(len(bboxs)):
            # Get location face
            x1, y1, x2, y2 = bboxs[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (204, 204, 255), 2)
            
            # Landmarks
            for x in range(5):
                point_x = int(landmarks[i][2 * x])
                point_y = int(landmarks[i][2 * x + 1])
                cv2.circle(frame, (point_x, point_y), tl+1, clors[x], -1)
            
            # Get face from location
            if isThread == True:
                isThread = False
                
                # Recognition
                face_image = frame[y1:y2, x1:x2]
                thread = Thread(target=recognition, args=(face_image,))
                thread.start()
        
            if name == null:
                continue
            else:
                if score < 0.4: # 유사도 0.6
                    caption= "UNKNOWN"
                    # print(caption)
                    unknown_cnt += 1 # Count Unknowns
    
                else:
                    now_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    recognition_res[name] = score, now_time
                    #print(recognition_res)
                    caption = f"{name.split('_')[0]}:{score:.2f}:{now_time}"
                    #print(caption)

                    path = './dataset/face-datasets'
                    file = make_dir_list(path)
                    for i in range(len(file)):
                        if file[i] in name :
                            #print(name)
                            #print(f"{name.split('_')[0]}:{score}:{now_time}")
                            recognition_res.update
                    
        # Count fps 
        frame_count += 1
        
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()
        
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Face Recognition", frame)
        
        if name in recognition_res :
            recognition_res

        #if name in recognition_res :
        #    print(recognition_res)
        #    print(name)
        if name in recognition_res :
            print(recognition_res)

        if cv2.waitKey(25) != -1 :
            break
    
if __name__=="__main__":
    server()