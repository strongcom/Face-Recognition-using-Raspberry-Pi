# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np

## TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## server ip, port
#s.connect(('192.168.1.108', 13300))
#s.connect(('192.168.0.28', 13000))
s.connect(('10.93.100.172', 13000))

## webcam 이미지 capture
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
if not cam.isOpened():
    print("Cannot open camera")
    exit()

## 이미지 속성 변경 3 = width, 4 = height
cam.set(3, 320);
cam.set(4, 240);

## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#print('check1')

while True:
    # time.sleep(5)
    # 비디오의 한 프레임씩 읽는다.
    # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
    ret, frame = cam.read()
#    print('check2')

    if ret == False :
        continue
    # cv2. imencode(ext, img [, params])
    # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.#
    #print('check3')
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    #print('check4')

    # frame을 String 형태로 변환
    data = np.array(frame)
    stringData = data.tobytes()

    #서버에 데이터 전송
    #(str(len(stringData))).encode().ljust(16)
    s.sendall((str(len(stringData))).encode().ljust(16) + stringData)

cam.release()