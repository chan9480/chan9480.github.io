---
title:  "wiki데이터를 이용한 이미지 캡셔닝"

categories:
  - pj
tags:
  - [attention, RNN]

toc: true
toc_sticky: true

date: 2022-03-28
last_modified_at: 2022-03-28
---


### mac gpu 사용가능확인 (1이면 사용가능)


```python
import tensorflow as tf
```


```python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

    Num GPUs Available:  0



```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


# import


```python
# tensorflow.compat.v2 를 사용
import tensorflow.compat.v2 as tf
import pandas as pd
from matplotlib import pyplot as plt
import re
import cv2
import time
```

# dataset
> 각 train 데이터 ( train_1 ~ train_10) 는 small_data.ipynb 에서 따로 처리하여 만들어짐.  
> 영어만 추출, image_url 과 caption_feature 만 추출.

출처 : https://www.kaggle.com/c/wikipedia-image-caption/code?competitionId=29705&searchQuery=tensor


```python
# caption_feature를 보면
# 1. [SEP] 으로 나뉘어져 있는데, 그 뒤에 있는 내용에 대해 할 것. (BERT)에서 사용하는 스페셜 토큰인데 파인튜닝은 안할 것이므로..
# 2. 숫자정보는 제외하자 대부분 특수한 경우에 쓰이거나 주소 등이다.

df = pd.read_csv('/content/drive/MyDrive/dataset/wiki/train_1.tsv', delimiter = '\t')
p = re.compile('\[SEP\].+') # \로 감싸진 곳은
df['caption_title_and_reference_description'] = df['caption_title_and_reference_description'].apply(
                                lambda x: '<start> ' +
                                        re.sub('\d+', '', p.search(x).group().replace('[SEP] ', '')).lower()
                                        +' <end>'
                                        if p.search(x).group() not in['[SEP] ', ''] else None)
df=df.dropna(axis=0)
print(df.shape)
df.head()
```

    (301484, 2)






  <div id="df-e23dee60-b000-4f2a-9687-a1c1871c9bd4">
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
      <th>image_url</th>
      <th>caption_title_and_reference_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://upload.wikimedia.org/wikipedia/commons...</td>
      <td>&lt;start&gt; downtown deer park &lt;end&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://upload.wikimedia.org/wikipedia/commons...</td>
      <td>&lt;start&gt; jürgen ovens's justitia, -, museumsber...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://upload.wikimedia.org/wikipedia/commons...</td>
      <td>&lt;start&gt;  mv agusta  raid &lt;end&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://upload.wikimedia.org/wikipedia/commons...</td>
      <td>&lt;start&gt; seth macfarlane's logo &lt;end&gt;</td>
    </tr>
    <tr>
      <th>6</th>
      <td>https://upload.wikimedia.org/wikipedia/commons...</td>
      <td>&lt;start&gt; erskine river at lorne &lt;end&gt;</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e23dee60-b000-4f2a-9687-a1c1871c9bd4')"
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
          document.querySelector('#df-e23dee60-b000-4f2a-9687-a1c1871c9bd4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e23dee60-b000-4f2a-9687-a1c1871c9bd4');
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
import cv2
import numpy as np
from urllib import request

def url_to_image(url):
    '''
    url 에서 이미지를 추출하여, (512,512,3) 의 rgb ndarray로 리턴
    '''
    resp = request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)#/255.0
    return image
    #return cv2.resize(image, (512,512))
```


```python
# 가장 긴 캡션과 그의 이미지 출력해보기 (모델상에서는 실행안해도됨.)
from google.colab.patches import cv2_imshow
long_caption = max(df['caption_title_and_reference_description'], key = lambda x: len(x))
a = df[df['caption_title_and_reference_description'] == long_caption ]['image_url']
#a = df[df['caption_title_and_reference_description'] == 'Downtown Deer Park']['image_url']
x = url_to_image(str(a.iloc[0]))
print(long_caption)
cv2_imshow(x)
print(x.shape)
print(len(long_caption))
```

    <start> figure : genomic context scheme of smrc and its closest homologues in other organisms. the αr rna genes are represented by red arrows and the flanking orfs by arrows on different colors depending on their product function (legend). numbers indicate the αr rna gene's and flanking orfs coordinates in each organism genome database. the gene strand is represented with the file direction. on the left of the figure identification names are used which correspond to a certain organism: αr_smrc = sinorhizobium meliloti  (nc_), αr_smedrc = sinorhizobium medicae wsm chromosome (nc_), αr_sfrc = sinorhizobium fredii ngr chromosome (nc_), αr_atrc = agrobacterium tumefaciens str. c chromosome linear (nc_), αr_reciatrc = rhizobium etli ciat  (nc_), αr_arrc = agrobacterium radiobacter k chromosome  (nc_), αr_rltrc = rhizobium leguminosarum bv. trifolii wsm (nc_), αr_avrc = agrobacterium vitis s chromosome  (nc_), αr_rlvrc = rhizobium leguminosarum bv. viciae  (nc_), αr_rltrc = rhizobium leguminosarum bv. trifolii wsm (nc_), αr_recfnrc = rhizobium etli cfn  (nc_), αr_mlrc = mesorhizobium loti maff chromosome (nc_), αr_mcrc = mesorhizobium ciceri biovar biserrulae wsm chromosome (nc_), αr_bcrcii = brucella canis atcc  chromosome ii (nc_), αr_bsrcii = brucella suis atcc  chromosome ii (nc_), αr_bmmrcii = brucella melitensis bv.  str. m chromosome ii (nc_), αr_basrcii = brucella abortus s chromosome  (nc_), αr_bmrcii = brucella melitensis atcc  chromosome ii (nc_), αr_bsrcii = brucella suis  chromosome ii (nc_), αr_barcii = brucella abortus bv.  str. - chromosome ii (nc_), αr_bmarcii = brucella melitensis biovar abortus  chromosome ii (nc_), αr_borcii = brucella ovis atcc  chromosome ii (nc_), αr_bmircii = brucella microti ccm  chromosome  (nc_), αr_oarc = ochrobactrum anthropi atcc  chromosome  (nc_), αr_msbncrc = mesorhizobium sp. bnc (nc_), αr_bahrc = bartonella henselae str. houston- (nc_), αr_bacrc = bartonella clarridgeiae  (nc_), αr_batrc = bartonella tribocorum cip  (nc_), αr_baqrc = bartonella quintana str. toulouse (nc_), αr_babrc = bartonella bacilliformis kc (nc_), αr_bagrc = bartonella grahamii asaup (nc_), αr_acrc = azorhizobium caulinodans ors  (nc_), αr_stnrc = starkeya novella dsm  chromosome (nc_), αr_xarc = xanthobacter autotrophicus py chromosome (nc_), αr_mesrc = methylocella silvestris bl chromosome (nc_), αr_beirc = beijerinckia indica subsp. indica atcc  chromosome (nc_), αr_rhprc = rhodopseudomonas palustris bisa chromosome (nc_). <end>



<img src = '/assets/images/project/source_2.png' width = '30%' height = '30%'>




    (7992, 1080, 3)
    2492


# 전처리
> 위에서 확인했듯, caption 이 제일 긴 행에 대해  
> 1. 이미지가 너무 크다(7992,1080). >  512,512 로 압축하게 되면 심각하게 찌그러 질것,
> 2. caption이 너무 길다. 3011자.

> 해결방법
> 1. caption이 너무 긴 행(100자 이상)은 삭제.  
> 2. image 파일이 너무 큰(가로 세로 비율이 2:1 혹은 1:2 를 초과) 경우는 삭제


```python
# 위 url_to_image 다시 정의
def url_to_image(url):
    '''
    url 에서 이미지를 추출하여, (512,512,3) 의 rgb ndarray로 리턴
    '''
    resp = request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype='uint8')
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)#/255.0
    if type(image) == type(None):
        return np.array([None])
    image = image/255.0
    if ( image.shape[0]/image.shape[1] > 2 ) or ( image.shape[0]/image.shape[1] < 1/2 ):  # 가로세로 비율이 1:2, 2:1을 벗어난다면
        return np.array([None])
    else:
        return cv2.resize(image, (299,299))
```


```python
X_pre_train=[]
y_train=[]
for i,j in enumerate((df.iloc[1500:2000]['image_url'])):
    try:
        temp = url_to_image(j)
    except:
        pass
    if None in temp:
        pass
    else:
        try:
          X_pre_train.append(temp)
          y_train.append( df.iloc[i]['caption_title_and_reference_description'] )
        except:
          pass
print(len(X_pre_train))
print(X_pre_train[2].shape)
print(len(y_train))
print(y_train[2])

# png 파일일 경우 libpng경고가 나오는데 무시해도 좋을듯하다.
# srv 파일의 경우
```

    388
    (299, 299, 3)
    388
    <start>  mv agusta  raid <end>



```python
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    87916544/87910968 [==============================] - 1s 0us/step
    87924736/87910968 [==============================] - 1s 0us/step



```python
# imagenet 가중치를 사용하여 특성추출
BATCH_SIZE = 64
image_dataset = tf.data.Dataset.from_tensor_slices(X_pre_train)
image_dataset = image_dataset.batch(BATCH_SIZE)
np_batch_features = np.array([np.zeros((64,2048))])   # 여기서 64는 batch_size가 아니라 InceptionV3의 마지막 레이어 아웃풋 모양
for img in image_dataset:
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))
    for temp_img in batch_features:
      np_batch_features = np.append(np_batch_features, np.array(temp_img).reshape(1,64,2048), axis=0)

np_batch_features = np_batch_features[1:]
```


```python
np_batch_features.shape
```




    (388, 64, 2048)




```python
# y_train 은 캡션 문장인데, 토크나이저를 통해 문장들을 단어별 벡터화 해준다. (cap_vector)

top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&*+.-;?@[]^`{}~ ')
tokenizer.fit_on_texts(y_train)
train_seqs = tokenizer.texts_to_sequences(y_train)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
print(y_train[3])
print(len(cap_vector))
cap_vector[3]
```

    <start> erskine river at lorne <end>
    388





    array([  2, 304,  31,   9, 305,   3,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=int32)




```python
#size_test = int( 0.01 * len(cap_vector))
size_test = 1       # test 는 딱 마지막에 확인용으로 사용하자.
X_test = np.array(np_batch_features)[0:size_test]
X_train = np.array(np_batch_features)[size_test:]
y_test = cap_vector[0:size_test]
y_train = cap_vector[size_test:]
len(X_test), len(y_test), len(X_train), len(y_train)
```




    (1, 1, 387, 387)




```python
X_test.shape
```




    (1, 64, 2048)



# 전역변수


```python
BUFFER_SIZE = 1000
embedding_dim = 512
units = 512
vocab_size = top_k + 1
num_steps = len(X_train) // BATCH_SIZE
features_shape = 2048
attention_features_shape = 64
```

## 데이터셋 지정


```python
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

# optimizer, loss func..


```python
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype = loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
```

# model 정의


```python
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    '''
    W1, W2, V 는 학습가능한 가중치벡터
    '''
    super(BahdanauAttention, self).__init__()   # 부모클래스 (tf.keras.Model.init()사용)
    self.W1 = tf.keras.layers.Dense(units)      # units(전역) 수 의 node를 갖는 W1 가중치
    self.W2 = tf.keras.layers.Dense(units)      # 위와 동 w2 가중치
    self.V = tf.keras.layers.Dense(1)           # V가중치는 하나로.

  def call(self, features, hidden):
    '''
    feautures : 이미지의 피쳐맵
    hidden : hidden state
    '''
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))
    score = self.V(attention_hidden_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights
```


```python
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
```


```python
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
```

# trian step 정의


```python
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
```


```python
checkpoint_path = "./drive/MyDrive/dataset/wiki/checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
```


```python
start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)
```


```python
# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []
```


```python
@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss
```


```python
EPOCHS = 20

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager.save()

    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
```

    Epoch 5 Batch 0 Loss 1.7554
    Epoch 5 Loss 1.809353
    Time taken for 1 epoch 157.57 sec

    Epoch 6 Batch 0 Loss 1.1847
    Epoch 6 Loss 1.441257
    Time taken for 1 epoch 46.38 sec

    Epoch 7 Batch 0 Loss 1.0604
    Epoch 7 Loss 1.354600
    Time taken for 1 epoch 45.60 sec

    Epoch 8 Batch 0 Loss 1.2198
    Epoch 8 Loss 1.334983
    Time taken for 1 epoch 45.94 sec

    Epoch 9 Batch 0 Loss 1.0771
    Epoch 9 Loss 1.477467
    Time taken for 1 epoch 46.14 sec

    Epoch 10 Batch 0 Loss 1.2247
    Epoch 10 Loss 1.347626
    Time taken for 1 epoch 45.70 sec

    Epoch 11 Batch 0 Loss 1.0726
    Epoch 11 Loss 1.260275
    Time taken for 1 epoch 46.71 sec

    Epoch 12 Batch 0 Loss 1.2278
    Epoch 12 Loss 1.200646
    Time taken for 1 epoch 46.94 sec

    Epoch 13 Batch 0 Loss 1.2482
    Epoch 13 Loss 1.216185
    Time taken for 1 epoch 46.13 sec

    Epoch 14 Batch 0 Loss 1.0949
    Epoch 14 Loss 1.133379
    Time taken for 1 epoch 45.98 sec

    Epoch 15 Batch 0 Loss 1.1599
    Epoch 15 Loss 1.292254
    Time taken for 1 epoch 46.25 sec

    Epoch 16 Batch 0 Loss 0.9927
    Epoch 16 Loss 1.213316
    Time taken for 1 epoch 46.76 sec

    Epoch 17 Batch 0 Loss 1.0226
    Epoch 17 Loss 1.102528
    Time taken for 1 epoch 46.03 sec

    Epoch 18 Batch 0 Loss 0.8826
    Epoch 18 Loss 1.238099
    Time taken for 1 epoch 46.58 sec

    Epoch 19 Batch 0 Loss 0.8167
    Epoch 19 Loss 1.090817
    Time taken for 1 epoch 47.73 sec

    Epoch 20 Batch 0 Loss 0.8715
    Epoch 20 Loss 1.050230
    Time taken for 1 epoch 48.49 sec



# 예측 모델


```python
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

max_length = max_length = calc_max_length(y_train)

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(url_to_image(image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
```


```python
# captions on the validation set
image_url = df['image_url'][20]
real_caption = df['caption_title_and_reference_description'][20]
result, attention_plot = evaluate(image_url)

cv2_imshow(url_to_image(image_url)*255)
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
```

<img src = '/assets/images/project/source_3.png' width='50%' height='50%'>



    Real Caption: <start> entering a mining settlement from jamieson <end>
    Prediction Caption: a np on either may river the wyndham's <end>

# 결과 해석
1. 광산 입구 길에 대한 이미지를 'wyndham의 river' 정도의 캡션을 생성했다.    
2. 길의 색깔만 다르다면 충분히 강으로 인식할 수 있는 수준일 것. 물론 성공적인 결과라고 보긴 힘들다.  

# 개선할 점
1. kaggle의 wiki학습데이터가 너무 방대하다 보니, 일부를 쪼개어 사용하기도 했고, 캡션의 경우 너무 디테일(지명이 들어간다던가 등)한 부분을 없애는 과정이 필요했다고 생각한다. (통계적 언어모델)  
2. 학습데이터의 증강이 더 필요했다고 생각한다. (색채 변경, 회전 등)  
3. 전체 구조를 좀 더 쪼개서 사용할 걸 싶다. 예를 들면 training.py 으로 가중치파일을 생성한다거나, main.py result.py 등 으로 하여 가시성, 접근성을 높이는 게 필요해보인다. (이 프로젝트는 몇달 전에 시도했던 것이기 떄문에 부족한점이 많이 보이는 듯 하다.)  

<br>

    🌜 개인 공부 기록용 블로그입니다. 오류나 틀린 부분이 있을 경우
    언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다! 😄

[맨 위로 이동하기](#){: .btn .btn--primary }{: .align-right}
