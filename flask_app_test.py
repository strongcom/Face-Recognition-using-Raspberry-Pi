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
from os import path
import revise
from revise import server

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def create_folder(path_img):
    if path.exists(path_img):
        shutil.rmtree(path_img)
    os.mkdir(path_img)

def save_images(images, path_img):
    for i, image in enumerate(images):
        save_img = Image.open(image)
        save_img.save(f"{path_img}/{i}.jpeg", format="JPEG")

def run_commands():
    output1 = subprocess.run(
        ["find", ".", "-name", "\"*.DS_Store\"", "-type", "f", "-delete"],
        capture_output=True,
    )
    print(output1.stdout.decode())

    output2 = subprocess.run(
        ["python3", "train.py", "--is-add-user=True"], capture_output=True
    )
    print(output2.stdout.decode())

@app.route("/image/userid", methods=["POST"])
def img():
    img = request.files.get("file")
    params = request.form
    path_img = f"./dataset/additional-training-datasets/{params['userid']}"
    create_folder(path_img)
    save_images(request.files.getlist("file"), path_img)
    run_commands()

    return "ok"

def send_result_to_external_api(user_id):
    url = "http://strongsumin.milk717.com/api/push/userId"
    data = {"userId": user_id}
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
    print(f"userId: {data['name']}")
    send_result_to_external_api(data["name"])

    return data["name"]

def run_flask_app():
    app.run(host="0.0.0.0", port=13330, debug=True, use_reloader=False)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.start()

    time.sleep(3)

    server_thread = threading.Thread(target=revise.server)
    server_thread.start()

    try:
        flask_thread.join()
        server_thread.join()
    except KeyboardInterrupt:
        print("Interrupted, closing threads.")
        server_thread.join()
        flask_thread.join()
