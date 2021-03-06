---
title:  "단순선형회귀, 회귀의 평가지표 그리고, 모델의 적합"

categories:
  - Something_else
tags:
  - []

toc: true
toc_sticky: true

date: 2022-01-21
last_modified_at: 2022-02-12
---

## 선형회귀
> 엑셀에서 점들의 추세선을 그어본적이 있다. 거기서 R값을 표시할 수도 있었는데, 이게 단순 선형회귀의 예시가 아닌가 싶다.  
> 이 또한, 머신러닝이다. 특정 평가지표가 최소가 되도록 모델에 학습데이터를 fit을 시켜주는 데, 평가지표의 종류는 아래와 같다.  
> y는 실제값(관측값, 측정값)     y^는 예측값 즉 ax+b, y평균은 y전체의 평균.  
> <img width="677" alt="image" src="https://user-images.githubusercontent.com/84547813/150494749-1be8815a-3e36-416c-9ff7-62370f5c540f.png">

~~~
# 라이브러리 import
from sklearn.linear_model import LinearRegression

# 데이터 정의
df = '타겟을 포함한 데이터 프레임'
df_test = '타겟 미포함 "테스트" 데이터 프레임'

# 모델 클래스 정의
model = LinearRegression()

# feature, target 정의
feature = ['feature_name']
target = ['target_name']
X_train = df[feature]
y_train = df[target]

# 모델 학습
model.fit(X_train, y_train)

# 예측값.
X_test = [[x] for x in df_test['feature_name']]
y_pred = model.predict(X_test)

# 계수확인
print('절편', model.intercept_)
print('계수(여러개일 수 있기때문에 array)', model.coef_)

# 시각화.
import matplotlib.pyplot as plt
## train 데이터에 대한 그래프를 검정색 점으로.
plt.scatter(X_train, y_train, color='black', linewidth=1)

## test 데이터에 대한 예측을 파란색 점으로.
plt.scatter(X_test, y_pred, color='blue', linewidth=1);
~~~

## 다중선형회귀
> x에 대해서 y의 추세선을 그으면 단순선형회귀, x와 y에 대해서 z의 추세선을 그으면 다중선형회귀가 될 것이다.   
> 그렇다면 4개이상 n개의 특성(feature)에 대해서 target의 추세선을 그릴 수 있을까?  
> 위 평가지표(MSE, MAE, R square)에서 예측값 y^ 는 ax+b일수도 있지만  
> ax+bw+c의 형태로도 될 수있고 일반화하면 아래와 같다.  
> <img width="102" alt="image" src="https://user-images.githubusercontent.com/84547813/150503177-644087c1-f1a0-4bb3-bc41-203cb552d7ec.png">

~~~
# 위 단순 선형회귀에서 X만 여러개 넣어주면 된다! (model도 동일)

# feature, target 정의
feature = ['feature_name_1', 'feature_name_2 ]
target = ['target_name']
X_train = df[feature]
y_train = df[target]

# 나머지는 동일!!
~~~

## 평가지표
> MSE (Mean Square Error),  
> MAE(Mean Absolute Error),  
> RMS or RMSE (Root Mean Square Error)  
> R-score

## 과적합 & 과소적합
**편향**은 잘못된 가정을 했을 때 발생하는 오차,  
**과소적합** 문제를 야기.  
**분산**은 트레이닝 셋의 복잡도에 의해 발생하는 오차, 큰 노이즈 까지 모델링에 포함시켜   
**과적합** 문제 야기.  
<img src="/assets/images/source_17.png" width="60%" height="60%" title="제목" alt="아무거나"/>

분산과 편향은 트레이드오프(trade-off)관계이다.  
예를 들어, 고분산 모델은 트레이닝 셋의 특성을 잘 담는다고도 할 수 있지만, 일반화에 실패했다고도 할 수 있다.  
고편향은 지나친 일반화를 하여 과소적합이 되는 것.  
즉 적절한 일반화는 편향과 분산의 적절한 분배를 의미하기도 한다.   
<img src="/assets/images/source_18.png" width="40%" height="40%" title="제목" alt="아무거나"/>

위 표에서 가로축은 모델의 복잡도, 세로축은 score 성능.  
즉, 가로축은 고편향에서 고분산으로 향한다.  

복잡도 up! -> 트레이닝셋 score up!  
너무 복잡해지면 고분산, 즉 과적합 ( 트레이닝셋에만 과하게 학습됨.)  
best model은 트레이닝셋과 독립적인 검증데이터에서 의 score로 찾아야 할 것.  

<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
