import time
from flask          import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import shutil
from PIL            import Image
from os             import path

import subprocess

app = Flask(__name__)  # Flask 클래스를 객체화시켜서 app이라는 변수에 저장


# create 함수를 end-point로 등록
@app.route("/image/userid", methods=['POST'])
def img(file=None) :
    #print(request.is_json)
    # username, image list 가 있는 json 

    params = request.get_json()

    # 해당 datasets을 넣을 폴더 생성
    path_img = './dataset/additional-training-datasets/' + params['username']

    # 이미 폴더가 있는 경우 해당 폴더 삭제
    if path.exists(path_img):
        shutil.rmtree(path_img)

    # 폴더 생성
    os.mkdir(path_img)
    
    # 이미지 저장
    img = params['image']
    for i in range(len(img)) :
        save_img = Image.open(img[i])
        save_img.save(path_img+'/'+ str(i) +".jpeg", format='JPEG')

    output1= subprocess.run(["find",".","-name","\"*.DS_Store\"","-type","f","-delete"], capture_output=True)
    print(output1.stdout.decode())
    output2 = subprocess.run(["python3", "train.py", "--is-add-user=True"], capture_output=True)
    print(output2.stdout.decode())
    return 'ok'

@app.route('/image/recognize', methods=['GET', 'POST']) 
def recognize() :
    process = subprocess.Popen(["python3", "server_.py"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT,
                               universal_newlines=True,
                               encoding='utf-8')
    while process.poll() == None:
        out = process.stdout.readline()
        print(out, end='')


'''
@app.route("/add_data", method=['POST'])
def add_data() :
    return 
'''

if __name__ == "__main__" :
    app.debug = True
    app.run(host='0.0.0.0', port=8080)