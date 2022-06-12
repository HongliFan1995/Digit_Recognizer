from tkinter import N
import tensorflow as tf
from sklearn import svm, neighbors

def CNN_model():
    """
    Define CNN model
    """
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

def SVM_model(kernel, C=1., degree=3, gamma='scale'):
    """
    Define SVM model
    """
    return svm.SVC(
        kernel=kernel,
        C = C,
        degree = degree,
        gamma = gamma,
        decision_function_shape='ovo',
    )

def KNN_model(n_neighbors=11):
    """
    Define KNN model
    """
    return neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)