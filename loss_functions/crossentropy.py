import tensorflow as tf


# 1. BINARY CROSSENTROPY
tf.keras.losses.BinaryCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="binary_crossentropy"
)

def binary_crossentropy(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred).numpy()


# 2. CATEGORICAL CROSSENTROPY
tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy"
)

def categorical_crossentropy(y_true, y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_true, y_pred).numpy()



if __name__ == "__main__":
    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0, 1, 0], [0, 0.01, 0.99]]

    print(categorical_crossentropy(y_true, y_pred))

    # w razie czego to wywołanie mozna fajnie zmodyfikować
