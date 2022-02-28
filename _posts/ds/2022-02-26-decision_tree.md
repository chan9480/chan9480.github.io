---
title:  "파이프라인(PipeLine)과 결정트리모델(Decision Tree)"

categories:
  - ds
tags:
  - []

toc: true
toc_sticky: true

date: 2022-02-26
last_modified_at: 2022-02-26
---

## 사이킷런(sklearn) 파이프라인 (PipeLine)
사이킷런에서 제공하는 파이프라인은 각 기능을 하는 모델들을 한번에 묶는 기능과  
하이퍼 파라미터를 연결시키는 기능이 있다.

~~~
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

pipe = make_pipeline(
    OneHotEncoder(use_cat_names=True),  
    SimpleImputer(),
    StandardScaler(),
    DecisionTreeClassifier(random_state=1, criterion='entropy', min_samples_leaf=10, max_depth=6)

    # min_samples_leaf는 말단 노드에 최소 존재해야할 데이터 수.
    # max_depth 는 최대 깊이를 제한하여 복잡도를 개선.
)

# pipe fit
pipe.fit(x_train, y_train)
print('검증세트 정확도', pipe.score(X_val, y_val))

# 테스트 셋
y_pred = pipe.predict(X_test)

# feature_importance 띄우기
## 먼저 파이프 내의 학습된 모델들 떼어서 가져온다.
model_dt = pipe.named_steps['decisiontreeclassifier']
enc = pipe.named_steps['onehotencoder']

encoded_columns = enc.transform(X_val).columns

importances = pd.Series(model_dt.feature_importances_, encoded_columns)
~~~

## 결정트리 (Decision_Tree)
결정트리는 '분류'에 있어서 마치 '회귀'의 선형회귀 와 같은 느낌이다.  
데이터들을 계속해서 두가지씩 분류하여 결과적으로 모든 데이터들을 정해진 갯수의 class들로 분류하게 된다.  

<img src="/assets/images/source_22.png" width="60%" height="60%" title="제목" alt="아무거나"/>  
결정트리를 발전시킨  
'랜덤포레스트 (Random_Forest)',  
'그래디언트 부스트 트리 (Gradient Boosted Tree)'  
같은 모델들을 더 많이 사용할 것이다.  
그러나 그 기초는 결정트리에 있다.  

> 트리학습에서의 비용함수

1. 지니지수 (Gini Impurity
2. 엔트로피 (Entropy)  

두가지 모두 불순도를 나타내는 척도 이며  
클수록 골고루 섞여 있다는 뜻.(10개의 공들 중에 5개씩 빨간공, 파란공이라면 0.5)
즉, 0에 가까울수록 치우쳐져 있다는 뜻. (전부 특정공만 10개 있다면 0)  

즉 지니지수, 엔트로피가 작아지는 방향으로 트리 노드를 생성한다.  



<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
