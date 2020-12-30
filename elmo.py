
import tensorflow_hub as hub
import tensorflow as tf

import re

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import plotly as py
import plotly.graph_objs as go

import warnings
warnings.filterwarnings("ignore")

global elmo
global embeddings

def init():
  global elmo
  elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

  global embeddings
  embeddings = elmo(
    sentences,
    signature="default",
    as_dict=True)["elmo"]


def get_words(text):
  sentences_dirty = text.split('.')

  max_len = 0

  sentences = []
  for sent in sentences_dirty:
      sent = re.sub('[,;-]', '', sent)
      if len(sent) > 0:
          sentences.append(sent)
          splitted = sent.split()
          if len(splitted) > max_len:
              max_len = len(splitted)

  words = []
  mask = []
  masked_words = []
  for sent in sentences:
    splitted = sent.split()
    for i in range(max_len):
      try:
        words.append(splitted[i])
      except:
        words.append('_')

  for word in words:
    if word == "_":
      mask.append(False)
    else:
      mask.append(True)
      masked_words.append(word)

  print('#########################')
  print(words)
  print('#########################')

  return sentences, mask

def get_embeddings(text):
  sentences, mask = get_words(text)




  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    x = sess.run(embeddings)
  embs = x.reshape(-1, 1024)
  masked_embs = embs[mask]

  return masked_embs



text = 'given strings var0, var1 . if length of var0 not equals to the length of var1 , return "NO". if both var0 and var1 contain at least one character "1" or both of them do not contain "1" at all , return "YES"; else return "NO" .'

masked_embs = get_embeddings(text)

print('#########################')
print(len(masked_embs))
print('#########################')

# PCA EDUCATIONAL - OR MAYBE ?? TO COMPRESS SENTENCES INTO POINTS OF LATENT SPACE ??
'''pca = PCA(n_components=10)
y = pca.fit_transform(masked_embs)

y = TSNE(n_components=2).fit_transform(y)



data = [
    go.Scatter(
        x=[i[0] for i in y],
        y=[i[1] for i in y],
        mode='markers',
        text=[i for i in masked_words],
    marker=dict(
        size=16,
        color = [len(i) for i in masked_words], #set color equal to a variable
        opacity= 0.8,
        colorscale='Viridis',
        showscale=False
    )
    )
]
layout = go.Layout()
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )
fig = go.Figure(data=data, layout=layout)
fig.show()'''


# Basic LSTM 
encoder = Sequential()
encoder.add(LSTM(16, input_shape=(1, 1024)))
encoder.add(Dense(16))
encoder.add(LSTM(16))
encoder.add(Dense(1))
encoder.compile(loss='mean_squared_error', optimizer='adam')
encoder.fit(x = masked_embs, epochs=100, batch_size=1, verbose=2)