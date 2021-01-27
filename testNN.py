from keras.models import load_model

print("START DATA LOAD")

X_test = np.load('data/X_train2.npy', allow_pickle=True)

Y_test = np.load('data/Y_train2.npy', allow_pickle=True)

print("DATA LOADED")

model = load_model('wizard_works1and2.h5')

print('MODEL LOADED')

result = model2.evaluate(X_test, Y_test)


print("RESULTS: ")
print("Loss on test: ", result[0])
print("Accuracy on test: ", result[1])


