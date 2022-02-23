---
title:  "신용카드 데이터로 k-means-cluster, pca 연습 "

categories:
  - pj
tags:
  - [cluster , pca]

toc: true
toc_sticky: true

date: 2022-02-21
last_modified_at: 2022-02-23
---


# 데이터 선정 :
kaggle credit customer dataset  (https://www.kaggle.com/arjunbhasin2013/ccdata)





```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
import pandas as pd
import numpy as np
df = pd.read_csv('/content/drive/MyDrive/dataset/CC GENERAL.csv.xls')
print(df.shape)
df.head()
```

    (8950, 18)






  <div id="df-0239163e-fb5c-444a-8b2b-314b7f2a8a42">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C10001</td>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C10002</td>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C10003</td>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C10005</td>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-0239163e-fb5c-444a-8b2b-314b7f2a8a42')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-0239163e-fb5c-444a-8b2b-314b7f2a8a42 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-0239163e-fb5c-444a-8b2b-314b7f2a8a42');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# 데이터 확인




```python
# 결측치 확인
df.isnull().sum()
```




    CUST_ID                               0
    BALANCE                               0
    BALANCE_FREQUENCY                     0
    PURCHASES                             0
    ONEOFF_PURCHASES                      0
    INSTALLMENTS_PURCHASES                0
    CASH_ADVANCE                          0
    PURCHASES_FREQUENCY                   0
    ONEOFF_PURCHASES_FREQUENCY            0
    PURCHASES_INSTALLMENTS_FREQUENCY      0
    CASH_ADVANCE_FREQUENCY                0
    CASH_ADVANCE_TRX                      0
    PURCHASES_TRX                         0
    CREDIT_LIMIT                          1
    PAYMENTS                              0
    MINIMUM_PAYMENTS                    313
    PRC_FULL_PAYMENT                      0
    TENURE                                0
    dtype: int64




```python
# 데이터 타입 확인
df.dtypes
```




    CUST_ID                              object
    BALANCE                             float64
    BALANCE_FREQUENCY                   float64
    PURCHASES                           float64
    ONEOFF_PURCHASES                    float64
    INSTALLMENTS_PURCHASES              float64
    CASH_ADVANCE                        float64
    PURCHASES_FREQUENCY                 float64
    ONEOFF_PURCHASES_FREQUENCY          float64
    PURCHASES_INSTALLMENTS_FREQUENCY    float64
    CASH_ADVANCE_FREQUENCY              float64
    CASH_ADVANCE_TRX                      int64
    PURCHASES_TRX                         int64
    CREDIT_LIMIT                        float64
    PAYMENTS                            float64
    MINIMUM_PAYMENTS                    float64
    PRC_FULL_PAYMENT                    float64
    TENURE                                int64
    dtype: object




```python
# 결측치 처리 (5%이하의 갯수이므로 드랍하겠다.)
df.dropna(inplace=True)
df.shape
```




    (8636, 18)




```python
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #경고문을 무시한다.

i=1
plt.figure(figsize= (20,40))
for col in df.drop('CUST_ID', axis=1).columns:
    plt.subplot(9,2,i)

    sns.distplot(df[col])

    i=i+1
plt.show()
```


<img src="/assets/images/source_11.png" width="60%" height="60%" title="제목" alt="아무거나"/>



```python
# 상관 계수 (correlation coefficient) 확인

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff838ba2890>




<img src="/assets/images/source_12.png" width="60%" height="60%" title="제목" alt="아무거나"/>



# 표준화 및 PCA
 +축소할 차원 수 결정.


```python
# 상관관계가 어느정도 있는 feature가 보이므로 PCA 사용이 유의미할 것으로 판단.
df.drop('CUST_ID', axis=1, inplace = True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# PCA전 표준화
ss= StandardScaler()
df= ss.fit_transform(df)

#PCA 진행
pca= PCA()
pca.fit(df)
```




    PCA()




```python
# PCA 의 축소 차원갯수에 대한 정확도 차이 (기존 차원수를 유지하는게 당연히 100%일것임..)
plt.plot(pca.explained_variance_ratio_.cumsum())
```




    [<matplotlib.lines.Line2D at 0x7ff8393998d0>]




<img src="/assets/images/source_13.png" width="60%" height="60%" title="제목" alt="아무거나"/>




```python
# 차원 축소 7개로 하겠다!
pca= PCA(n_components=7)
df_pca= pca.fit_transform(df)
```


```python
df_pca.shape
```




    (8636, 7)



# K-means Clustering
 +K-means 와 실루엣 스코어(silhouette_score)

  + k-means : 각 데이터들과 해당 centroid까지 거리 합
  + sil_score : 1에 가까울수록 군집과 군집이 잘 분리되어있다는 뜻  
  기본적으로 0이상이고 만약 음수라면 군집끼리 겹쳤다는 의미


```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.figure(figsize=(15,10))
distortions=[]
sil_scores=[]
for i in range(2,30):
    # n_cluster : 군집 갯수, n_iter : 중심점 업데이트의 최소 횟수
    kmeans= KMeans(n_clusters=i, n_init=10, init= 'k-means++', algorithm='full', max_iter=300)
    kmeans.fit(df_pca)
    # inertia_ : k-means 구하는 중에 centroid로부터 데이터들의 거리 데이터 (클 수록 중심점으로부터 멀다는 거겟쥐)
    distortions.append(kmeans.inertia_)
    label= kmeans.labels_
    sil_scores.append(silhouette_score(df_pca, label))
plt.plot(np.arange(2,30,1), distortions, alpha=0.5)
plt.plot(np.arange(2,30,1), distortions,'o' ,alpha=0.5)
plt.show()
```


<img src="/assets/images/source_14.png" width="60%" height="60%" title="제목" alt="아무거나"/>




```python
plt.figure(figsize=(15,10))
# sil_scores 확인
plt.plot(np.arange(2,30,1), sil_scores)
plt.plot(np.arange(2,30,1), sil_scores,'o' ,alpha=1)
plt.show()
```


<img src="/assets/images/source_15.png" width="60%" height="60%" title="제목" alt="아무거나"/>




```python
sil_scores
# k-means 는 군집을 늘릴수록 감소 (이상적)
# 실루엣 계수는 3에서 0.28정도로 그나마 크나, 전체적으로는 0.25근처로 별로 크게 나오진 않음.
```




    [0.24692450845604266,
     0.28065750721461125,
     0.2507007863496503,
     0.2408710781955889,
     0.25789671860000646,
     0.25543152274952236,
     0.26596171359898996,
     0.25980372706530475,
     0.2606025124276636,
     0.24195602184083095,
     0.25144013095228146,
     0.23962807370519182,
     0.23713138793264468,
     0.23981648640294187,
     0.2199707245532794,
     0.2259075081873923,
     0.23446181289456078,
     0.21832255569020825,
     0.24002990422966186,
     0.23109689399081063,
     0.22381807371604162,
     0.21453215289991348,
     0.21537478705286658,
     0.21174420647359052,
     0.2161573025700017,
     0.21208142071199876,
     0.2196422007478946,
     0.20723949847328374]




```python
# k-means 는 군집 3개로(실루엣계수 따라), PCA는 2개로하여 시각화.

kmeans= KMeans(n_clusters=3, n_init=10, init= 'k-means++', algorithm='full', max_iter=300)
kmeans.fit(df_pca)
labels= kmeans.labels_

pca= PCA(n_components=2)
temp = pca.fit_transform(df)
df_pca2 = pd.DataFrame(data=temp, columns=['pca1','pca2'])
df_pca2['labels']= labels
df_pca2.head()
```





  <div id="df-ed39b30d-40d0-4339-9588-39e85728f5f0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pca1</th>
      <th>pca2</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.696397</td>
      <td>-1.122594</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.215688</td>
      <td>2.435597</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.935860</td>
      <td>-0.385170</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.614640</td>
      <td>-0.724592</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.223706</td>
      <td>-0.783584</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ed39b30d-40d0-4339-9588-39e85728f5f0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ed39b30d-40d0-4339-9588-39e85728f5f0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ed39b30d-40d0-4339-9588-39e85728f5f0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize=(10,10))
ax = sns.scatterplot(x='pca1', y='pca2', hue='labels', data=df_pca2, palette='bright')
```


<img src="/assets/images/source_16.png" width="60%" height="60%" title="제목" alt="아무거나"/>




```python

```
<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
