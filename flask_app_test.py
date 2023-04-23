import threading
import time
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import shutil
from PIL import Image
from os import path
import json
import revise

import subprocess

from revise import server

# Flask app
app = Flask(__name__)  # Flask 클래스를 객체화시켜서 app이라는 변수에 저장
app.config['JSON_AS_ASCII'] = False

# create 함수를 end-point로 등록
@app.route("/image/userid", methods=['POST'])
def img():
    # form-data로 전달된 파일을 받기 위해 request.files 사용
    # 'file'은 클라이언트 측에서 파일을 전송할 때 사용한 필드명
    img = request.files.get('file')
    params = request.form

    # 해당 datasets을 넣을 폴더 생성
    path_img = './dataset/additional-training-datasets/' + params['userid']

    # 이미 폴더가 있는 경우 해당 폴더 삭제
    if path.exists(path_img):
        shutil.rmtree(path_img)

    # 폴더 생성
    os.mkdir(path_img)

    # 이미지 저장
    for i, image in enumerate(request.files.getlist('file')):
        save_img = Image.open(image)
        save_img.save(path_img + '/' + str(i) + '.jpeg', format='JPEG')

    output1 = subprocess.run(["find", ".", "-name", "\"*.DS_Store\"", "-type", "f", "-delete"], capture_output=True)
    print(output1.stdout.decode())

    output2 = subprocess.run(["python3", "train.py", "--is-add-user=True"], capture_output=True)
    print(output2.stdout.decode())

    return 'ok'


@app.route('/image/result', methods=['POST'])
def result():
    data = request.get_json()
    print("Name: {}".format(data['name']))
    # print("Name: {}\nScore: {}\nTime: {}".format(data['name'], data['score'], data['time']))
    return data['name']


import threading
import time
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import shutil
from PIL import Image
from os import path
from flask import Flask, jsonify
from flask import Flask, request, jsonify
import json
import revise

import subprocess

from revise import server

# Flask app
app = Flask(__name__)  # Flask 클래스를 객체화시켜서 app이라는 변수에 저장
app.config['JSON_AS_ASCII'] = False

# create 함수를 end-point로 등록
@app.route("/image/userid", methods=['POST'])
def img():
    # form-data로 전달된 파일을 받기 위해 request.files 사용
    # 'file'은 클라이언트 측에서 파일을 전송할 때 사용한 필드명
    img = request.files.get('file')
    params = request.form

    # 해당 datasets을 넣을 폴더 생성
    path_img = './dataset/additional-training-datasets/' + params['userid']

    # 이미 폴더가 있는 경우 해당 폴더 삭제
    if path.exists(path_img):
        shutil.rmtree(path_img)

    # 폴더 생성
    os.mkdir(path_img)

    # 이미지 저장
    for i, image in enumerate(request.files.getlist('file')):
        save_img = Image.open(image)
        save_img.save(path_img + '/' + str(i) + '.jpeg', format='JPEG')

    output1 = subprocess.run(["find", ".", "-name", "\"*.DS_Store\"", "-type", "f", "-delete"], capture_output=True)
    print(output1.stdout.decode())

    output2 = subprocess.run(["python3", "train.py", "--is-add-user=True"], capture_output=True)
    print(output2.stdout.decode())

    return 'ok'


@app.route('/image/result', methods=['POST'])
def result():
    data = request.get_json()
    print("Name: {}".format(data['name']))
    # print("Name: {}\nScore: {}\nTime: {}".format(data['name'], data['score'], data['time']))
    return data['name']


if __name__ == '__main__':
    def run_flask_app():
        app.run(host='0.0.0.0', port=13330, debug=True, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    time.sleep(3)  # Give the Flask app time to start

    server_thread = threading.Thread(target=revise.server)
    server_thread.start()

    try:
        flask_thread.join()
        server_thread.join()
    except KeyboardInterrupt:
        print("Interrupted, closing threads.")
        server_thread.join()
        flask_thread.join()