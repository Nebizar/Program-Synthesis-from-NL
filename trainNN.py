import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

print("START DATA LOAD")

X_train = np.load('data/X_train1.npy', allow_pickle=True)

Y_train = np.load('data/Y_train1.npy', allow_pickle=True)

print("DATA LOADED")

model = Sequential()
#Encoder
model.add(LSTM(100, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(50, activation='relu'))
#Decoder/Generator
model.add(RepeatVector(Y_train.shape[1]))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1850, activation='softmax')))
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.save('wizard_bare')

print("STARTING TRAINING")
model.fit(X_train, Y_train, epochs=10, verbose=0)
print("TRAINING FINISHED")

model.save('wizard_done')