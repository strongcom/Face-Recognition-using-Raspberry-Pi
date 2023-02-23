# Face Recognition 
[Reference] https://github.com/vectornguyen76/face-recognition.git


# 환경
* MacOS
* conda, Homebrew 설치, Jupyter notebook 가능

# 테스트 전 진행
https://fringe-singer-dff.notion.site/AI-Face-Recognition-Code-Test-11cb5630d4f7479296606d8a4x6e1e18f


# 테스트
* 아래 링크에서 각각 폴더에 있는 파일 다운로드 후 지정 파일로 이동
    - https://drive.google.com/drive/folders/1XT-WlFuI_nNbjd5KjXCYnGFSZ8HDsnul?usp=share_link
    - resnet -> insightface
    - yolov5 -> yolov5_face 

    <img width="311" alt="스크린샷 2023-02-23 오후 4 49 47" src="https://user-images.githubusercontent.com/76231561/220849482-66f33ac4-6b47-4537-ad38-b16c2a33ee41.png">
<br>

* 사람 추가 한 후 테스트할 경우
1. ./dataset/additional-training-datasets/ 폴더 아래 <name> 폴더를 생성한 후 사진 넣기
2. 아래 코드 실행
```
python3 train.py --is-add-user=True
python3 recognize.py
```

* "*.DS_Store" 관련 에러 발생 시 
```
find . -name "*.DS_Store" -type f -delete
```