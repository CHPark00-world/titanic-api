# Titanic Survival Prediction API

캐글 타이타닉 데이터로 학습한 모델을 FastAPI로 서비스하는 프로젝트.

## 결과

- 캐글 점수: 0.787
- 모델: RandomForest (max_depth=5), 교차검증 0.83

## 한 것

- 결측치 처리 (Age → 호칭별 중앙값, Embarked → 최빈값)
- 이름에서 호칭(Mr/Mrs/Miss/Master) 추출 → +0.5%p
- RandomForest / GradientBoosting / LogisticRegression 비교
- joblib로 모델 저장 후 FastAPI `/predict` 엔드포인트로 제공

## 실행

pip install -r requirements.txt
python train.py # train.csv 필요 (캐글에서 다운로드)
python -m uvicorn main:app --reload

# http://127.0.0.1:8000/docs
