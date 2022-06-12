import tensorflow as tf
from sklearn import svm

def CNN_model():

    inp = tf.keras.Input(shape=(28,28,))

    x = tf.keras.layers.Reshape((28,28,1))(inp)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outp = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs = inp, outputs = outp)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer = opt,
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy'],
    )

    return model

def SVM_model(kernel, degree=3, C=1):
