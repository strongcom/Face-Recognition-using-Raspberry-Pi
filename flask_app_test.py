import threading
import time
import os
import shutil
import json
import subprocess
import requests
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename
import server_revise
from server_revise import server
import os
from flask import Flask, request
from PIL import Image
import shutil
import subprocess

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def create_folder(path_img):
    if not os.path.exists(path_img):  # 폴더가 이미 존재하는지 확인
        os.mkdir(path_img)

def save_images(images, path_img):
    for i, image in enumerate(images):
        try:
            save_img = Image.open(image)
            save_img.save(os.path.join(path_img, f"{i}.jpeg"), format="JPEG")
        except IOError:
            # 이미지 저장 실패 시 예외 처리
            print(f"Failed to save image {i}")

def run_commands():
    output1 = subprocess.run(
        ["find", ".", "-name", "*.DS_Store", "-type", "f", "-delete"],
        capture_output=True,
        text=True
    )
    print(output1.stdout)

    output2 = subprocess.run(
        ["python3", "train.py", "--is-add-user=True"],
        capture_output=True,
        text=True
    )
    print(output2.stdout)

@app.route("/image/userid", methods=["POST"])
def img():
    userid = request.form.get('userid')  # None이 아닌 값으로 설정하도록 변경
    if not userid:
        return "Invalid request: userid is missing", 400

    path_img = os.path.abspath(f"./dataset/additional-training-datasets/{userid}")
    create_folder(path_img)

    images = request.files.getlist("file")
    if not images:
        return "Invalid request: no images were uploaded", 400

    save_images(images, path_img)
    run_commands()

    return "ok"

def send_result_to_external_api(user_id, time):
    url = "http://strongsumin.milk717.com/api/push/userId"
    data = {
        "userId": user_id,
        "time": time
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        print("Result sent to external API")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Error occurred: {err}")

@app.route("/image/result", methods=["POST"])
def result():
    data = request.get_json()
    print("userId: "+ data['name'] + " / time: " + data["time"])
    send_result_to_external_api(data["name"], data["time"])

    return jsonify({"name": data["name"], "time": data["time"]}), 200

def run_flask_app():
    app.run(host="0.0.0.0", port=13330, debug=True, use_reloader=False)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    time.sleep(3)

    server_thread = threading.Thread(target=server_revise.server)
    server_thread.start()

    try:
        flask_thread.join()
        server_thread.join()
    except KeyboardInterrupt:
        print("Interrupted, closing threads.")
        server_thread.join()
        flask_thread.join()
