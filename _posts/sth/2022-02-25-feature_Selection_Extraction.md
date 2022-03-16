---
title:  "특성 선택과 특성 추출"

categories:
  - Something_else
tags:
  - []

toc: true
toc_sticky: true

date: 2022-02-25
last_modified_at: 2022-03-16
---

차원을 축소하는 데에는 어떤 방법이 있을까.  
첫번째로 영향도 높은 특성을 **'고르는 것'** 과  
두번째는 특성 모두를 특정 축에 **'투영시키는 것'** 이 있다.

# 특성 선택 (feature selection)
특성을 줄이는데,   
영향도가 큰 특성을 골라야  
모델의 정확도를 유지하면서 복잡도를 줄이거나 일반화를 할 수 있을 것이다.  
## 1. Filter method (전처리 과정에서 통계값으로 선택.)  
통계값 종류 : 카이제곱, ANOVA_f_score, 상관계수 등.  
~~~
from sklearn.feature_selection import chi2, SelectKBest
selector1 = SelectKBest(chi2, k=14330)
X_train1 = selector1.fit_transform(X_train, y_train)
X_test1 = selector1.transform(X_test)
~~~
↪️ 카이제곱으로 선택한 예시  
## 2. Wrapper method (모델학습과 검정(validation)을 반복하면서 특성을 선택)  
'무식한 정답' 이지만 비합리적인 방법.(greedy 나 grid 단어가 떠오른다.. 왜 발음도 비슷하지ㅎ)  
시간과 비용이 많이 소모되며, 상대적으로 filter metthod 보다 과적합되기 쉽다.  
forward step, backward step, stepwise 방식이 있는데,  
각각 feature를 하나씩 '추가', '소거', '추가소거 병합' 방법이다.  
~~~
from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=15,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4)
features = feature_selector.fit(np.array(X, y)
filtered_features= train_features.columns[list(features.k_feature_idx_)]
#이후 filtered_features로 모델에 fit
~~~
↪️ mlxtend 라이브러리의 SequentialFeatureSelector 를 사용하여, forward step예시  
roc_auc 를 평가지표로, RandomForestClassifier 모델을 이용하여 forward step 기법을 사용.  
forward 변수만 False로 바꾸면 backward로 사용  
~~~
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
           min_features=2,
           max_features=4,
           scoring='roc_auc',
           print_progress=True,
           cv=2)
features = feature_selector.fit(X, y)
filtered_features= train_features.columns[list(features.k_feature_idx_)]
#이후 filtered_features로 모델에 fit
~~~
↪️ mlxtend의 ExhaustiveFeatureSelector를 사용한 stepwise 예시  
추가소거를 동시에 하는 기법, 최종feature의 최소갯수와 최대갯수를 정한다.  
## 3. Embedded method (학습과 동시에)
l1 norm(LASSO회귀에서 사용, 절댓값으로 loss함수 보정),  
l2 norm(RIDGE회귀에서 사용, 제곱값으로 loss함수 보정),  
elastic_net(l1 + l2),  
selectfrommodel(sklearn의 함수를 이용하여, decision Tree기반 모델, 로지스틱등의 모델 등을 통해 feature_importance계산하여 feature 선택)  
~~~
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model  import LogisticRegression

# 3가지 모델을 이용해서 feature_importance를 통한 feature선택 모델 설정.
RFselector = SelectFromModel(estimator=RandomForestClassifier()).fit(X, y)
GBMselector = SelectFromModel(estimator=GradientBoostingClassifier()).fit(X, y)
LRselector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)

# feature 보기 (get_support 내부변수를 보면 True, False로 선정 feature를 구별한다)
columns = data.columns
RF_selected = columns[RFselector.get_support()]
GBM_selected = columns[GBMselector.get_support()]
LR_selected = columns[LRselector.get_support()]
~~~

# 특성 추출 (feature extraction)
대표적으로 pca와 같이 특성들을   
특정 축( 특성들의 특징을 잘 나타내야 한다는 이유로 보통 분산을 최대로하는 축을 선택한다. )에  
투영(projection)시켜 차원을 축소하는 방식이 있다.  
방법에 대해서는 <https://chan9480.github.io/ds/pca/> 에서 조금 다루어놓았다.  

한계점을 말하자면
무조건 직선에 투영시킨다는 것.
분산이 큰 특성을 무조건 중요 특성으로 판단한다는 것.
# 그래서 차원축소... 선택할거야, 추출할거야?
답은 없겠지만 개인적으로 pca는 시각화나 비지도 군집화에 유용하게 사용되며,  
그 외 분류, 회귀 등에는 선택이 많이 쓰인다고 봐도 될 것 같다.  
그 중에서도 filter method와 embedded method 의 응용을 앞으로 좀 더 공부해보고자 한다.  


<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
