---
title:  "비행기 푯값 예측 MLP(Multi Layer Perceptron)"

categories:
  - pj
tags:
  - [MLP]

toc: true
toc_sticky: true

date: 2022-03-26
last_modified_at: 2022-03-26
---

# 판다스 프로파일링 코랩에서 작동하도록 버전지정 설치

```python
!pip install pandas==1.2.1
!pip install pandas_profiling==2.8.0
```

    Collecting pandas==1.2.1
      Downloading pandas-1.2.1-cp37-cp37m-manylinux1_x86_64.whl (9.9 MB)
    [K     |████████████████████████████████| 9.9 MB 9.8 MB/s
    [?25hRequirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas==1.2.1) (2018.9)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas==1.2.1) (2.8.2)
    Requirement already satisfied: numpy>=1.16.5 in /usr/local/lib/python3.7/dist-packages (from pandas==1.2.1) (1.21.5)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas==1.2.1) (1.15.0)
    Installing collected packages: pandas
      Attempting uninstall: pandas
        Found existing installation: pandas 1.3.5
        Uninstalling pandas-1.3.5:
          Successfully uninstalled pandas-1.3.5
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    google-colab 1.0.0 requires requests~=2.23.0, but you have requests 2.27.1 which is incompatible.[0m
    Successfully installed pandas-1.2.1




    Requirement already satisfied: pandas_profiling==2.8.0 in /usr/local/lib/python3.7/dist-packages (2.8.0)
    Requirement already satisfied: jinja2>=2.11.1 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (2.11.3)
    Requirement already satisfied: missingno>=0.4.2 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (0.5.1)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (1.21.5)
    Requirement already satisfied: htmlmin>=0.1.12 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (0.1.12)
    Requirement already satisfied: visions[type_image_path]==0.4.4 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (0.4.4)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (1.0.1)
    Requirement already satisfied: pandas!=1.0.0,!=1.0.1,!=1.0.2,>=0.25.3 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (1.2.1)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (1.7.3)
    Requirement already satisfied: phik>=0.9.10 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (0.12.2)
    Requirement already satisfied: tangled-up-in-unicode>=0.0.6 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (0.1.0)
    Requirement already satisfied: matplotlib>=3.2.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (3.2.2)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (2.27.1)
    Requirement already satisfied: tqdm>=4.43.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (4.63.0)
    Requirement already satisfied: confuse>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (1.7.0)
    Requirement already satisfied: astropy>=4.0 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (4.3.1)
    Requirement already satisfied: ipywidgets>=7.5.1 in /usr/local/lib/python3.7/dist-packages (from pandas_profiling==2.8.0) (7.7.0)
    Requirement already satisfied: attrs>=19.3.0 in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.4.4->pandas_profiling==2.8.0) (21.4.0)
    Requirement already satisfied: networkx>=2.4 in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.4.4->pandas_profiling==2.8.0) (2.6.3)
    Requirement already satisfied: imagehash in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.4.4->pandas_profiling==2.8.0) (4.2.1)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from visions[type_image_path]==0.4.4->pandas_profiling==2.8.0) (7.1.2)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from astropy>=4.0->pandas_profiling==2.8.0) (4.11.3)
    Requirement already satisfied: pyerfa>=1.7.3 in /usr/local/lib/python3.7/dist-packages (from astropy>=4.0->pandas_profiling==2.8.0) (2.0.0.1)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.7/dist-packages (from confuse>=1.0.0->pandas_profiling==2.8.0) (6.0)
    Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.5.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (3.6.0)
    Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.2.0)
    Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.10.1)
    Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.1.1)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.2.0)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.5.1->pandas_profiling==2.8.0) (1.1.0)
    Requirement already satisfied: tornado>=4.0 in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.1.1)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.3.5)
    Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.4.2)
    Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (2.6.1)
    Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (1.0.18)
    Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.8.1)
    Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (57.4.0)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.7.5)
    Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2>=2.11.1->pandas_profiling==2.8.0) (2.0.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.0->pandas_profiling==2.8.0) (1.4.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.0->pandas_profiling==2.8.0) (3.0.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.0->pandas_profiling==2.8.0) (0.11.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=3.2.0->pandas_profiling==2.8.0) (2.8.2)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib>=3.2.0->pandas_profiling==2.8.0) (3.10.0.2)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from missingno>=0.4.2->pandas_profiling==2.8.0) (0.11.2)
    Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.3.3)
    Requirement already satisfied: jupyter-core in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.9.2)
    Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.4.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.18.1)
    Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.7/dist-packages (from importlib-resources>=1.4.0->jsonschema!=2.5.0,>=2.4->nbformat>=4.2.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (3.7.0)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas!=1.0.0,!=1.0.1,!=1.0.2,>=0.25.3->pandas_profiling==2.8.0) (2018.9)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.2.5)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (1.15.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->pandas_profiling==2.8.0) (2021.10.8)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->pandas_profiling==2.8.0) (2.10)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->pandas_profiling==2.8.0) (1.24.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.7/dist-packages (from requests>=2.23.0->pandas_profiling==2.8.0) (2.0.12)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.3.1)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.13.3)
    Requirement already satisfied: nbconvert in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (5.6.1)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (1.8.0)
    Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.7/dist-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (22.3.0)
    Requirement already satisfied: ptyprocess in /usr/local/lib/python3.7/dist-packages (from terminado>=0.8.1->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.7.0)
    Requirement already satisfied: PyWavelets in /usr/local/lib/python3.7/dist-packages (from imagehash->visions[type_image_path]==0.4.4->pandas_profiling==2.8.0) (1.3.0)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (1.5.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.7.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.4)
    Requirement already satisfied: bleach in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (4.1.0)
    Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.6.0)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.8.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (21.3)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.5.1->pandas_profiling==2.8.0) (0.5.1)


# 데이터 불러오기

```python
# 데이터 불러오기
import pandas as pd
from pandas_profiling import ProfileReport
import datetime
CSV_PATH_1 = '/content/drive/MyDrive/dataset/airplane/Data_Train.csv'
CSV_PATH_2 = '/content/drive/MyDrive/dataset/airplane/Test_set.csv'
CSV_PATH_3 = '/content/drive/MyDrive/dataset/airplane/Sample_submission.csv'
target = 'Price'
```

# 데이터 엔지니어

```python
import re

# feature 수 10개.
df_train = pd.read_csv(CSV_PATH_1)
X_train = df_train.drop(target, axis=1)
y_train = df_train[target]
X_test = pd.read_csv(CSV_PATH_2)
y_test = pd.read_csv(CSV_PATH_3)
"""
전처리
1. Airline : one-hot encoding
2. Date_of_Journey : 삭제
3. Source : one-hot encoding
4. Destination : one-hot encoding
5. Route : 삭제하자
6. Dep_Time : h*60 + min 으로 치환
7. Arrival_Time : 위와 동
8. Duration : 위와동
9. Total_Stops : 숫자만 남기자 non-stop은 0
10. Additional_Info : one-hot encoding
"""

# Route  Date_of_journey 삭제
X_train=X_train.drop(['Date_of_Journey'], axis=1)
X_test=X_test.drop(['Date_of_Journey'], axis=1)
X_train=X_train.drop(['Route'], axis=1)
X_test=X_test.drop(['Route'], axis=1)

# 탑승, 도착 시간계산
def hour_cal(a):
    hour = int( re.findall(r'(\d+):', a)[0] )
    minute = int( re.findall(r':(\d+)', a)[0] )
    return hour*60+minute
X_train['Dep_Time']=X_train['Dep_Time'].apply(hour_cal)
X_test['Dep_Time']=X_test['Dep_Time'].apply(hour_cal)
X_train['Arrival_Time']=X_train['Arrival_Time'].apply(hour_cal)
X_test['Arrival_Time']=X_test['Arrival_Time'].apply(hour_cal)

# 소요시간 계산
def hour_cal2(a):
    try:
        hour = int( re.findall(r'(\d+)h', a)[0] )
    except:
        hour=0
    try:
        minute = int( re.findall(r'(\d+)m', a)[0] )
    except:
        minute = 0
    return hour*60+minute
X_train['Duration']=X_train['Duration'].apply(hour_cal2)
X_test['Duration']=X_test['Duration'].apply(hour_cal2)

# stop에서 숫자만 남기자
X_train['Total_Stops']=X_train['Total_Stops'].apply(lambda x: 0 if str(x)[0]=='n' else int(str(x)[0]))
X_test['Total_Stops']=X_train['Total_Stops'].apply(lambda x: 0 if str(x)[0]=='n' else int(str(x)[0]))
print(X_train.info())

X_train.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10683 entries, 0 to 10682
    Data columns (total 8 columns):
     #   Column           Non-Null Count  Dtype
    ---  ------           --------------  -----
     0   Airline          10683 non-null  object
     1   Source           10683 non-null  object
     2   Destination      10683 non-null  object
     3   Dep_Time         10683 non-null  int64
     4   Arrival_Time     10683 non-null  int64
     5   Duration         10683 non-null  int64
     6   Total_Stops      10683 non-null  int64
     7   Additional_Info  10683 non-null  object
    dtypes: int64(4), object(4)
    memory usage: 667.8+ KB
    None






  <div id="df-5e346ee7-84fc-40d0-874c-f1a960b92b39">
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
      <th>Airline</th>
      <th>Source</th>
      <th>Destination</th>
      <th>Dep_Time</th>
      <th>Arrival_Time</th>
      <th>Duration</th>
      <th>Total_Stops</th>
      <th>Additional_Info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>1340</td>
      <td>70</td>
      <td>170</td>
      <td>0</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Air India</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>350</td>
      <td>795</td>
      <td>445</td>
      <td>2</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jet Airways</td>
      <td>Delhi</td>
      <td>Cochin</td>
      <td>565</td>
      <td>265</td>
      <td>1140</td>
      <td>2</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>3</th>
      <td>IndiGo</td>
      <td>Kolkata</td>
      <td>Banglore</td>
      <td>1085</td>
      <td>1410</td>
      <td>325</td>
      <td>1</td>
      <td>No info</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IndiGo</td>
      <td>Banglore</td>
      <td>New Delhi</td>
      <td>1010</td>
      <td>1295</td>
      <td>285</td>
      <td>1</td>
      <td>No info</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5e346ee7-84fc-40d0-874c-f1a960b92b39')"
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
          document.querySelector('#df-5e346ee7-84fc-40d0-874c-f1a960b92b39 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5e346ee7-84fc-40d0-874c-f1a960b92b39');
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
df_train = X_train
df_train['price'] = y_train
df_train.profile_report()
```


    Output hidden; open in https://colab.research.google.com to view.

# 인코딩 (one hot encoding 사용)

```python
# 원핫 인코딩

#X_train=pd.get_dummies(data = X_train, columns = ['Airline', 'Source', 'Destination', 'Total_Stops',  'Additional_Info'])
#X_test=pd.get_dummies(data = X_test, columns = ['Airline', 'Source', 'Destination', 'Total_Stops',  'Additional_Info'])


temp = pd.concat([X_train, X_test])
temp = pd.get_dummies(data = temp, columns = ['Airline', 'Source', 'Destination', 'Total_Stops',  'Additional_Info'])
X_train_encoded = temp[0:len(X_train)]
X_test_encoded = temp[len(X_train):]

print(X_train_encoded.info())
print(X_test_encoded.info())
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10683 entries, 0 to 10682
    Data columns (total 42 columns):
     #   Column                                        Non-Null Count  Dtype  
    ---  ------                                        --------------  -----  
     0   Dep_Time                                      10683 non-null  int64  
     1   Arrival_Time                                  10683 non-null  int64  
     2   Duration                                      10683 non-null  int64  
     3   price                                         10683 non-null  float64
     4   Airline_Air Asia                              10683 non-null  uint8  
     5   Airline_Air India                             10683 non-null  uint8  
     6   Airline_GoAir                                 10683 non-null  uint8  
     7   Airline_IndiGo                                10683 non-null  uint8  
     8   Airline_Jet Airways                           10683 non-null  uint8  
     9   Airline_Jet Airways Business                  10683 non-null  uint8  
     10  Airline_Multiple carriers                     10683 non-null  uint8  
     11  Airline_Multiple carriers Premium economy     10683 non-null  uint8  
     12  Airline_SpiceJet                              10683 non-null  uint8  
     13  Airline_Trujet                                10683 non-null  uint8  
     14  Airline_Vistara                               10683 non-null  uint8  
     15  Airline_Vistara Premium economy               10683 non-null  uint8  
     16  Source_Banglore                               10683 non-null  uint8  
     17  Source_Chennai                                10683 non-null  uint8  
     18  Source_Delhi                                  10683 non-null  uint8  
     19  Source_Kolkata                                10683 non-null  uint8  
     20  Source_Mumbai                                 10683 non-null  uint8  
     21  Destination_Banglore                          10683 non-null  uint8  
     22  Destination_Cochin                            10683 non-null  uint8  
     23  Destination_Delhi                             10683 non-null  uint8  
     24  Destination_Hyderabad                         10683 non-null  uint8  
     25  Destination_Kolkata                           10683 non-null  uint8  
     26  Destination_New Delhi                         10683 non-null  uint8  
     27  Total_Stops_0                                 10683 non-null  uint8  
     28  Total_Stops_1                                 10683 non-null  uint8  
     29  Total_Stops_2                                 10683 non-null  uint8  
     30  Total_Stops_3                                 10683 non-null  uint8  
     31  Total_Stops_4                                 10683 non-null  uint8  
     32  Additional_Info_1 Long layover                10683 non-null  uint8  
     33  Additional_Info_1 Short layover               10683 non-null  uint8  
     34  Additional_Info_2 Long layover                10683 non-null  uint8  
     35  Additional_Info_Business class                10683 non-null  uint8  
     36  Additional_Info_Change airports               10683 non-null  uint8  
     37  Additional_Info_In-flight meal not included   10683 non-null  uint8  
     38  Additional_Info_No Info                       10683 non-null  uint8  
     39  Additional_Info_No check-in baggage included  10683 non-null  uint8  
     40  Additional_Info_No info                       10683 non-null  uint8  
     41  Additional_Info_Red-eye flight                10683 non-null  uint8  
    dtypes: float64(1), int64(3), uint8(38)
    memory usage: 813.7 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2671 entries, 0 to 2670
    Data columns (total 42 columns):
     #   Column                                        Non-Null Count  Dtype  
    ---  ------                                        --------------  -----  
     0   Dep_Time                                      2671 non-null   int64  
     1   Arrival_Time                                  2671 non-null   int64  
     2   Duration                                      2671 non-null   int64  
     3   price                                         0 non-null      float64
     4   Airline_Air Asia                              2671 non-null   uint8  
     5   Airline_Air India                             2671 non-null   uint8  
     6   Airline_GoAir                                 2671 non-null   uint8  
     7   Airline_IndiGo                                2671 non-null   uint8  
     8   Airline_Jet Airways                           2671 non-null   uint8  
     9   Airline_Jet Airways Business                  2671 non-null   uint8  
     10  Airline_Multiple carriers                     2671 non-null   uint8  
     11  Airline_Multiple carriers Premium economy     2671 non-null   uint8  
     12  Airline_SpiceJet                              2671 non-null   uint8  
     13  Airline_Trujet                                2671 non-null   uint8  
     14  Airline_Vistara                               2671 non-null   uint8  
     15  Airline_Vistara Premium economy               2671 non-null   uint8  
     16  Source_Banglore                               2671 non-null   uint8  
     17  Source_Chennai                                2671 non-null   uint8  
     18  Source_Delhi                                  2671 non-null   uint8  
     19  Source_Kolkata                                2671 non-null   uint8  
     20  Source_Mumbai                                 2671 non-null   uint8  
     21  Destination_Banglore                          2671 non-null   uint8  
     22  Destination_Cochin                            2671 non-null   uint8  
     23  Destination_Delhi                             2671 non-null   uint8  
     24  Destination_Hyderabad                         2671 non-null   uint8  
     25  Destination_Kolkata                           2671 non-null   uint8  
     26  Destination_New Delhi                         2671 non-null   uint8  
     27  Total_Stops_0                                 2671 non-null   uint8  
     28  Total_Stops_1                                 2671 non-null   uint8  
     29  Total_Stops_2                                 2671 non-null   uint8  
     30  Total_Stops_3                                 2671 non-null   uint8  
     31  Total_Stops_4                                 2671 non-null   uint8  
     32  Additional_Info_1 Long layover                2671 non-null   uint8  
     33  Additional_Info_1 Short layover               2671 non-null   uint8  
     34  Additional_Info_2 Long layover                2671 non-null   uint8  
     35  Additional_Info_Business class                2671 non-null   uint8  
     36  Additional_Info_Change airports               2671 non-null   uint8  
     37  Additional_Info_In-flight meal not included   2671 non-null   uint8  
     38  Additional_Info_No Info                       2671 non-null   uint8  
     39  Additional_Info_No check-in baggage included  2671 non-null   uint8  
     40  Additional_Info_No info                       2671 non-null   uint8  
     41  Additional_Info_Red-eye flight                2671 non-null   uint8  
    dtypes: float64(1), int64(3), uint8(38)
    memory usage: 203.5 KB
    None

# 특성 선택 (SelectKBest사용)

```python
from sklearn.feature_selection import chi2, SelectKBest
selector1 = SelectKBest(chi2, k=5)
X_train1 = selector1.fit_transform(X_train_encoded, y_train)
columns = X_train_encoded.columns
X_train_encoded = X_train_encoded[columns[selector1.get_support()]]
X_test_encoded = X_test_encoded[columns[selector1.get_support()]]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-33-6ef92159fc82> in <module>()
          1 from sklearn.feature_selection import chi2, SelectKBest
          2 selector1 = SelectKBest(chi2, k=5)
    ----> 3 X_train1 = selector1.fit_transform(X_train_encoded, y_train)
          4 columns = X_train_encoded.columns
          5 X_train_encoded = X_train_encoded[columns[selector1.get_support()]]


    /usr/local/lib/python3.7/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
        853         else:
        854             # fit method of arity 2 (supervised transformation)
    --> 855             return self.fit(X, y, **fit_params).transform(X)
        856
        857


    /usr/local/lib/python3.7/dist-packages/sklearn/feature_selection/_univariate_selection.py in fit(self, X, y)
        405             )
        406
    --> 407         self._check_params(X, y)
        408         score_func_ret = self.score_func(X, y)
        409         if isinstance(score_func_ret, (list, tuple)):


    /usr/local/lib/python3.7/dist-packages/sklearn/feature_selection/_univariate_selection.py in _check_params(self, X, y)
        604             raise ValueError(
        605                 "k should be >=0, <= n_features = %d; got %r. "
    --> 606                 "Use k='all' to return all features." % (X.shape[1], self.k)
        607             )
        608


    ValueError: k should be >=0, <= n_features = 4; got 5. Use k='all' to return all features.

# 스케일링 및 모델 구성, fit

```python
from sklearn.preprocessing import StandardScaler
# StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_train_scaled=pd.DataFrame(data=X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test_encoded)
X_test_scaled=pd.DataFrame(data=X_test_scaled)
```


```python
# modeling MLP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=4, kernel_initializer='he_normal'))
model.add(Dense(128, activation='tanh', kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l1(0),   
                activity_regularizer=regularizers.l2(0))
         )
Dropout(0.3)
model.add(Dense(128, activation='sigmoid', kernel_initializer='he_normal'))
Dropout(0.5)
model.add(Dense(1, activation='linear'))

model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

# model fit
history = model.fit(X_train_scaled, y_train, batch_size = 256, validation_split=0.2, epochs=4000, verbose=0)
```

# loss 변동 그래프로 확인

```python
import matplotlib.pyplot as plt
val_loss = history.history['val_loss']
val_mae = history.history['val_mae']
epochs = range(1, len(val_loss) + 1)


plt.plot(epochs, val_loss, 'b', label='val loss')

plt.show()
```


<img src = '/assets/images/project/source_1.png' width = '90%' height ='90%'>

> ..? 너무 모양 완벽한 그래프를 갖는다.. 왜지

# 어떤 예측값을 갖지...?


```python
model.predict(X_test_encoded[0:10])
```




    array([[9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562],
           [9079.562]], dtype=float32)

```python
# mae가 최소로하는 하나의 값으로 수렴해버림..
# 딥러닝까지 필요없는 데이터에 딥러닝을 써버리면..
# 복잡도만 올라가고 성능은 나오지 않는 것 같다.
```

# 소잡는칼...!

<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
