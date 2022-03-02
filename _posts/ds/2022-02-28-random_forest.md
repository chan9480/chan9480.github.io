---
title:  "랜덤포레스트(Random Forest)와 ..."

categories:
  - ds
tags:
  - []

toc: true
toc_sticky: true

date: 2022-02-28
last_modified_at: 2022-02-28
---

## 랜덤 포레스트 모델
단순 선형회귀와 릿지(Ridge) 회귀가 있었다면,  
결정트리와 랜덤포레스트가 있다.  

릿지 회귀에서 과적합을 방지하는 장치가 있었다.  
랜덤포레스트 또한 결정트리에서 더 일반화시켜주는 장치가 있다.  

결정트리를 여러개 만들어 모두의 의견을 합산하여 판단을 내린다.  

> 숲(포레스트)이 만들어지는 과정  

1. 많은 feature중에서 n개 (pram: max_features) 를 '랜덤'으로 고른다.  
2. n개의 feature중 가장 영향도가 큰 feature를 골라 첫번째 node를 생성하고, 나머지 feature 중 랜덤하게 골라 트리를 완성한다.  
3. 위와같은 트리를 m개(pram: n_estimators)만든다.  
4. 트리들의 분류 결과로 투표를 해서 최종 결정을 한다.  

~~~
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# bootstrap을 False로 하고 max_features 수정가능.
classifier = RandomForestClassifier(n_estimators = 50, bootstrap = False, max_features = 5)

# 모델 fit
classifier.fit(X_train, y_train)

# 결과값 예측
y_pred = classifier.predict(X_test)

# 같은지 다른지 확인.
print("정확도 : {}".format(accuracy_score(y_test, y_pred))
~~~

> CV(cross validation)은 GridsearchCV, RandomizedSearchCV 등을 이용.  

~~~
# CV(cross validation)은 gridsearch 등을 이용.
from sklearn.model_selection import GridSearchCV

grid = {
    'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,7,10],
    'min_samples_split' : [2,3,5,10]
}

classifier_grid = GridSearchCV(classifier, param_grid = grid, scoring="accuracy", n_jobs=-1, verbose =1)

classifier_grid.fit(X_train, y_train)

print("최고 평균 정확도 : {}".format(classifier_grid.best_score_))
print("최고의 파라미터 :", classifier_grid.best_params_)
~~~
> feature_importances 내부변수로 확인가능  

~~~
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

feature_importances = model.feature_importances_

ft_importances = pd.Series(feature_importances, index = X_train.columns)
ft_importances = ft_importances.sort_values(ascending=False)

plt.figure(fig.size(12,10))
sns.barplot(x=ft_importances, y= X_train.columns)
plt.show()
~~~



<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
