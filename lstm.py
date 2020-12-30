from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Basic LSTM 
encoder = Sequential()
encoder.add(LSTM(16, input_shape=(1, 1024)))
encoder.add(Dense(16))
encoder.add(LSTM(16))
encoder.add(Dense(1))
encoder.compile(loss='mean_squared_error', optimizer='adam')
#encoder.fit(masked_embs, _, epochs=100, batch_size=1, verbose=2)