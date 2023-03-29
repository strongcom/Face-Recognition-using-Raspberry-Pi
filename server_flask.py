from flask import Flask, request, jsonify
# jsonify ; dictionary -> JSON 변환
# request ; http request 가능
import server_

import server_

print(server_.server())

app = Flask(__name__)  # Flask 클래스를 객체화시켜서 app이라는 변수에 저장

'''
# create 함수를 end-point로 등록
@app.route("/mkdir_train", methods=['POST'])
def mkdir_train() :
    #print(request.is_json)
    params = request.get_json()
    username = params['username']
    userid = params['userid']
'''

@app.route('/test', methods=['POST']) 
def recognize_name() :
    return server_.server()

'''
@app.route("/add_data", method=['POST'])
def add_data() :
    return 
'''

if __name__ == "__main__" :
    app.debug = True
    app.run(host='', port=8080)