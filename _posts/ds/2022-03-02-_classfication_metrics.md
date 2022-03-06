---
title:  "이진분류의 평가지표"

categories:
  - Something_else
tags:
  - []

toc: true
toc_sticky: true

date: 2022-03-02
last_modified_at: 2022-03-02
---
## 이진분류 모델의 종류
로지스틱 회귀,  
Decision_Tree(base model),  
RandomForest,  
Gradient_boost,  
KNN(K-Nearest-Neighbors),  
그외 딥러닝 모델도 될 수 있겠다.  

## 평가지표의 종류
위와 같은 이진분류 모델 (군집x, 지도학습의 분류o) 의 평가지표는 어떤 게 있을까.  
(회귀에서는 MSE, MAE, RMS, R-score 등이 있었쥬!)  

가장 단순하게 볼 수 있는 것은 validation 데이터에 대해서 얼마나 많이 맞춘 비율(0~1)이 있을 것이다.  
바로 정확도(Accuracy)!    
그러나 모델을 하나의 값으로 평가하면 안될 것이다.    

> 정확도 (Accuracy)  
먼저 정확도를 체크할 때에도 베이스 모델은 반드시 필요하다.  
그 이유를 예로 들어보자면,  암을 예측하는 모델의 학습데이터에서 암에 걸린 데이터(1)가 5%, 나머지 95%가 건강(0)하다면,  
입력에 상관없이 항상 출력을 '0'으로만 한다면 그 모델은  
CV나 hold-out 으로 비슷한 비율로 생성된 validation 데이터에서도 정확도를 대략 0.95로 가질 것이다..!  
이러한 모델은 정확도 검증에 있어서 0.95는 최소한 넘는 것으로 고려해야 할 것이다.  

> TP, TN, FP, FN  
(T,F는 True, False,   P,N은 Positive, Negative,)  
True는 맞춘거, False는 못맞춘거.  
P와 N은 예측한 거 기준. (FP는 실제 Negative인데 모델이 Positive로 예측한거야)  

> 정밀도(precision)와 재현율(recall, sensitivity), specificity(TN-rate), Fall-out  
precision = TP/(TP+FP) = Positive로 예측한 것들중 맞춘비율  
recall = TP/(TP+FN) = 실제 Positive 중 맞춘비율  (TPRate)
specificity = TN/(TN+FP) = 실제 Negative 중 맞춘비율 (TNR)
Fall-out = FP/(FP+TN) = 실제 Negative 중 틀린비율 (FPR)

> confusion Metrics  
TP, TN, FP, FN 을 한번에 나타낸 행렬  
아래 사진을 참고하면 한번에 이해될것.ㅎㅎ  

~~~
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# pipe는 파이프라인
fig, ax = plt.subplots()
matrix = plot_confusion_matrix(pipe, X_val_cleaned, y_val,
                            cmap = plt.cm.Blues,
                            ax = ax);
~~~
<img src="/assets/images/source_23.png" width="30%" height="30%" title="제목" alt="아무거나"/>  


> AUC, ROC 와 임계값(Threshold)  
ROC 커브는 임계값에 따른 FPR과 TPR 값이 그리는 커브인데. 모두 일대일 관계이다.   
AOC수치는 ROC curve의 아래 면적을 뜻한다.  
AOC는 클수록 좋은 모델인 것은 사실이다.  
이유 : FPR (틀린비율, 낮을수록 좋음) 이 낮음에도 TPR(맞춘비율, 높을수록 좋음)은 높아야 AOC가 높은 것이기 때문.  
결과적으로 tpr-fpr이 최대가 되는 점의 임계점을 고르는게 최선이라고 할 수 있다.  

<img src="/assets/images/source_24.png" width="30%" height="30%" title="제목" alt="아무거나"/>  
<img src="/assets/images/source_25.png" width="30%" height="30%" title="제목" alt="아무거나"/>  



<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
