# Face Recognition 
[Reference] https://github.com/vectornguyen76/face-recognition.git


# 기본 환경
* MacOS
    ; conda, Homebrew 설치 완료, Jupyter notebook 설치 완료
* Windows
    ; CMAKE, Dlib 설치 완료, Jupyter notebook 설치 완료

# 환경 구축
* [MacOS](https://fringe-singer-dff.notion.site/AI-Face-Recognition-11cb5630d4f7479296606d8a46e1e18f)
* [Windows](https://fringe-singer-dff.notion.site/AI-Windows-3707e5de34a64c3d82ef844c08787141)


# 테스트
<<<<<<< HEAD
* 아래 링크에서 각각 폴더에 있는 파일 다운로드 후 지정 파일로 이동
    - https://drive.google.com/drive/folders/1QBnN_as3ShQ_TRbTitFleP_lOEoXk8oB?usp=share_link
    - resnet -> insightface
    - yolov5 -> yolov5_face 
=======
* 아래 링크에서 파일 다운로드 후 지정 파일로 이동 <br>
    - [model](https://drive.google.com/drive/folders/1QBnN_as3ShQ_TRbTitFleP_lOEoXk8oB?usp=sharing)
    - resnet100_backbone.pth -> insightface
    - yolov5n-0.5.pt -> yolov5_face 
<br><br>

* 라즈베리파이 연결한 후 flask_app_test.py를 실행

* "*.DS_Store" 관련 에러 발생 시 
```
find . -name "*.DS_Store" -type f -delete
```


* 이전에는 로컬에서 라즈베리파이 카메라와 서버를 연결하는 방식이었으나, 현재는 라즈베리파이에서 외부 네트워크에 있는 서버로 값을 전송해주는 방식으로 변경
