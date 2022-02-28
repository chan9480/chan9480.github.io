---
title:  "Ridge 회귀와 로지스틱 회귀"

categories:
  - ds
tags:
  - []

toc: true
toc_sticky: true

date: 2022-02-25
last_modified_at: 2022-02-25
---

## CV (교차 검증, cross validation)
먼저 데이터라고 하면 용도에 따라 3가지로 분류할 수 있다.  
학습데이터(training_data),  
검증데이터(validation_data),  
테스트데이터(test_data)  

세가지 데이터는 서로 누출되어서는 안되고,  
학습데이터로 학습시킨 '모델' 의  
점수는 '검증데이터'로 체크를 하되,  
마지막으로 타겟 데이터로써 테스트 데이터를 사용한다.  

단, CV는 '검증데이터'를 따로 두지 않고,  
학습데이터를 일부 추출하여 검증데이터로 사용한다는 방식이다.

만약 그저 데이터를 처음부터 '학습데이터'와 '검증데이터'로 나눈다면 아래와 같을 것이다.  
<img src="/assets/images/source_19.png" width="60%" height="60%" title="제목" alt="아무거나"/>

그러나 CV를 이용하게 되면 아래와 같이 학습데이터와 검증데이터가 수시로 바뀌게 되며. 더 **'일반화'** 된 모델을 얻을 수 있다.  
<img src="/assets/images/source_20.png" width="60%" height="60%" title="제목" alt="아무거나"/>


## 릿지 회귀 (Ridge regression)
간단하게 Ridge 회귀는 단순회귀보다 더 일반화된 모델을 만든다고 이해해도 좋을 듯 하다.  
즉, 단순선형회귀의 과적합 방지 모델.  


~~~
from sklearn.linear_model import RidgeCV

# 수행해볼 알파 값들을 정의한다.
alphas = [0.01, 0.05, 0.1, 0.2, 1.0, 10.0, 100.0]

# RidgeCV 모델 객체를 정의한다.
ridge = RidgeCV(alphas=alphas, normalize=True, cv=3)
ridge.fit(ans[['x']], ans['y'])

#결정된 알파와 베스트 스코어를 출력.
print("alpha :", ridge.alpha_)
print("best score :", ridge.best_score_)

# 예측해보고 싶은 X_test에 대해 y
y_pred = ridge.predict(X_test)
~~~

## 로지스틱 회귀 (logistic Regression)
로지스틱 회귀는 sigmoid라는 '비선형'을 사용한 '2진 분류(classification)' 모델이다.  

예시 ) 환자들의 생체 데이터 + 암에 걸리지 않았다면 0, 걸렸다면 1을 갖는 feature (target)
~~~
from sklearn.linear_model import LogisticRegression

model = LogisticRegressionCV(penalty="l1", Cs=[1.0], solver='liblinear', cv=3)
model.fit(X, y)
# 여기서 X는 여러 feature를 갖고 있을 수 있고,
# y는 0이나 1의 값을 갖는 feature일 것이다!

# 스코어 프린트
print('best score: ', model.scores_)

# 테스트 데이터 확인
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test) # proba는 y_pred를 결정했던 근거가 된 각 확률을 보여준다.

# 계수 확인
coefficients = pd.Series(model.coef_[0], X.columns)
coefficients    # 계수가 -1부터 1사이의 값을 갖는데 -1에 가까운 값일 수록 0이 나오게 하는 feature임을 의미
~~~

> sigmoid

x값이 어떤값이든지, y값은 0부터 1사이의 값을 비선형으로 갖도록 하는 함수
<img src="/assets/images/source_21.png" width="60%" height="60%" title="제목" alt="아무거나"/>

> 임계값 (Classification Threshold)

임계값이 로지스틱 회귀에서 등장했는데,  
어떤 데이터 하나가 A에 속할 확률이 0.41이 나왔다고 했을때,  
임계값이 0.5(default)라면 당연히 'A가 아니다' 라고 하겠지만,  
임계값을 0.4으로 조정한다면 'A 이다' 라고 판단을 내리게 된다.  

<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
