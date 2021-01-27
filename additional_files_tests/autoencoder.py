from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
#from keras.utils import plot_model

import elmo
from naps.pipelines.read_naps import read_naps_dataset
from dataset import flatten, generate_output, get_tokens, even_embeddings, even_tokens
from tree_transformation import make_the_tree_good
from loss_function import init_loss, loss_function

# text = 'given the number var0 . var7 is new array of strings . create an array var1 with size var0 + 1 of type boolean . for var4 in range from 2 to var0 ( inclusive ) if var1 [ var4 ] continue , set var5 to var4 + var4 , while var5 is less than or equal to var0 repeat you have to store true in var1 [ var5 ] and then increase var5 by var4 , let var6 : = var4 , while var0 is greater than or equal to var6 repeat increment var2 , concatenate concatenation of string value of var6 and space to var3 and then let var6 : = var4 * var6 . append string value of var2 to the end of var7 . add var3 to var7 . you have to return var7'
# out_tokens = ['func', 'char**', '__main__', 'var', 'int', 'var0', 'var', 'bool*', 'var1', 'var', 'int', 'var2', 'var', 'char*', 'var3', 'var', 'int', 'var4', 'var', 'int', 'var5', 'var', 'int', 'var6', 'var', 'char**', 'var7', 'assign', 'char**', 'var', 'char**', 'var7', 'invoke', 'char**', '_ctor', 'assign', 'bool*', 'var', 'bool*', 'var1', 'invoke', 'bool*', '_ctor', 'invoke', 'int', '+', 'var', 'int', 'var0', 'val', 'int', 1, 'assign', 'int', 'var', 'int', 'var2', 'val', 'int', 0, 'assign', 'char*', 'var', 'char*', 'var3', 'val', 'char*', '', 'assign', 'int', 'var', 'int', 'var4', 'val', 'int', 2, 'while', 'void', 'invoke', 'bool', '<=', 'var', 'int', 'var4', 'var', 'int', 'var0', 'if', 'void', 'invoke', 'bool', 'array_index', 'var', 'bool*', 'var1', 'var', 'int', 'var4', 'continue', 'void', 'assign', 'int', 'var', 'int', 'var5', 'invoke', 'int', '+', 'var', 'int', 'var4', 'var', 'int', 'var4', 'while', 'void', 'invoke', 'bool', '<=', 'var', 'int', 'var5', 'var', 'int', 'var0', 'assign', 'bool', 'invoke', 'bool', 'array_index', 'var', 'bool*', 'var1', 'var', 'int', 'var5', 'val', 'bool', True, 'assign', 'int', 'var', 'int', 'var5', 'invoke', 'int', '+', 'var', 'int', 'var5', 'var', 'int', 'var4', 'assign', 'int', 'var', 'int', 'var6', 'var', 'int', 'var4', 'while', 'void', 'invoke', 'bool', '<=', 'var', 'int', 'var6', 'var', 'int', 'var0', 'assign', 'int', 'var', 'int', 'var2', 'invoke', 'int', '+', 'var', 'int', 'var2', 'val', 'int', 1, 'assign', 'char*', 'var', 'char*', 'var3', 'invoke', 'char*', 'concat', 'var', 'char*', 'var3', 'invoke', 'char*', 'concat', 'invoke', 'char*', 'str', 'var', 'int', 'var6', 'val', 'char*', ' ', 'assign', 'int', 'var', 'int', 'var6', 'invoke', 'int', '*', 'var', 'int', 'var6', 'var', 'int', 'var4', 'assign', 'int', 'var', 'int', 'var4', 'invoke', 'int', '+', 'var', 'int', 'var4', 'val', 'int', 1, 'invoke', 'void', 'array_push', 'var', 'char**', 'var7', 'invoke', 'char*', 'str', 'var', 'int', 'var2', 'assign', 'char**', 'var', 'char**', 'var7', 'invoke', 'char**', 'array_concat', 'var', 'char**', 'var7', 'invoke', 'char**', 'string_split', 'var', 'char*', 'var3', 'val', 'char*', ' \\t', 'return', 'void', 'var', 'char**', 'var7']


# elmo.init()
# masked_embs = elmo.get_embeddings(text)

# # print(len(out_tokens))
# # print(masked_embs.shape)

# unique = get_tokens()
# #print(unique)

# # # reshape input into [samples, timesteps, features]
# n_out = len(out_tokens)
# masked_embs = masked_embs.reshape((1, masked_embs.shape[0], masked_embs.shape[1]))#dataset size, embeddings(evened), 1024 (elmo)
# output_true = generate_output(out_tokens, unique)
# output_true = output_true.reshape((1, output_true.shape[0], output_true.shape[1]))
# print(masked_embs.shape)
# print(output_true.shape)
# # define model
# model = Sequential()
# #Encoder
# model.add(LSTM(100, activation='relu', input_shape=(masked_embs.shape[1], masked_embs.shape[2]), return_sequences=True))
# model.add(LSTM(50, activation='relu'))
# #Decoder/Generator
# model.add(RepeatVector(n_out))
# model.add(LSTM(50, activation='relu', return_sequences=True))
# model.add(LSTM(100, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(1764, activation='softmax')))
# model.compile(optimizer='adam', loss='mse')

# # fit model
# model.fit(masked_embs, output_true, epochs=10, verbose=0)
# #check if ok
# print(model.get_weights())

# # demonstrate recreation
# yhat = model.predict(sequence, verbose=0)
# print(yhat[0,:,0])

def prepare_train_data():
    elmo.init()
    unique = get_tokens()

    ds, _, test = read_naps_dataset()
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    Y_tokens = []
    Y_tokens2 = []
    max_embs_len = 0
    max_tokens_len = 0
    with ds:
        for d in ds:
            if "is_partial" in d and d["is_partial"]:
                continue
            masked_embs = elmo.get_embeddings(' '.join(d["text"]))
            if len(masked_embs) > max_embs_len:
                max_embs_len = len(masked_embs)
            X_train.append(masked_embs)    
            tokens = make_the_tree_good(d["code_tree"]["funcs"])
            #print(make_the_tree_good(d["code_tree"]["funcs"]))
            if len(tokens) > max_tokens_len:
                max_tokens_len = len(tokens)
            Y_tokens.append(tokens)
            break

    with test:
        for t in test:
            if "is_partial" in t and t["is_partial"]:
                continue
            masked_embs = elmo.get_embeddings(' '.join(t["text"]))
            if len(masked_embs) > max_embs_len:
                max_embs_len = len(masked_embs)
            X_test.append(masked_embs)    
            tokens = make_the_tree_good(t["code_tree"]["funcs"])
            #print(make_the_tree_good(d["code_tree"]["funcs"]))
            if len(tokens) > max_tokens_len:
                max_tokens_len = len(tokens)
            Y_tokens2.append(tokens)
            break

    X_train_2 = []
    #print(X_train)
    for ex in X_train:
        X_train_2.append(even_embeddings(ex, max_embs_len))
    
    for ex in Y_tokens:
        #print(even_tokens(ex,max_tokens_len))
        Y_train.append(generate_output(even_tokens(ex,max_tokens_len), unique))

    X_test_2 = []
    #print(X_train)
    for ex in X_test:
        X_test_2.append(even_embeddings(ex, max_embs_len))
    
    for ex in Y_tokens2:
        #print(even_tokens(ex,max_tokens_len))
        Y_test.append(generate_output(even_tokens(ex,max_tokens_len), unique))

    return array(X_train_2), array(Y_train), array(X_test_2), array(Y_test) 

def train_model():
    init_loss()
    X_train, Y_train, X_test, Y_test = prepare_train_data()

    model = Sequential()
    #Encoder
    model.add(LSTM(100, activation='sigmoid', input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True))
    model.add(LSTM(50, activation='sigmoid'))
    #Decoder/Generator
    model.add(RepeatVector(Y_test.shape[1]))
    model.add(LSTM(50, activation='sigmoid', return_sequences=True))
    model.add(LSTM(100, activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(1850, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit model
    model.fit(X_train, Y_train, epochs=10, verbose=1, batch_size = 32)
    print(model.get_weights())
    print(model.summary())

# x, y, x2, y2 = prepare_train_data()
# print(x.shape)
# print(y.shape)
# print(x2.shape)
# print(y2.shape)

train_model()