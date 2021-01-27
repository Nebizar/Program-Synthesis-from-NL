import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import load_model

print("START DATA LOAD")

X_train = np.load('data/X_train2.npy', allow_pickle=True)

Y_train = np.load('data/Y_train2.npy', allow_pickle=True)

print("DATA LOADED")

#model = load_model('wizard_done.h5')

#print('MODEL LOADED')

model2 = Sequential()
#Encoder
model2.add(LSTM(100, activation='sigmoid', input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True))
model2.add(LSTM(50, activation='sigmoid'))
#Decoder/Generator
model2.add(RepeatVector(Y_test.shape[1]))
model2.add(LSTM(50, activation='sigmoid', return_sequences=True))
model2.add(LSTM(100, activation='sigmoid', return_sequences=True))
model2.add(TimeDistributed(Dense(1850, activation='softmax')))
#model2.add(GaussianNoise(0.5))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.save('wizard_bare')

print("STARTING TRAINING")
model2.fit(X_test, Y_test, epochs=10, verbose=1, batch_size = 32)
print("TRAINING FINISHED")

model2.save('wizard_works1and2.h5')