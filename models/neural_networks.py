import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras


def neural_networks(games_df, x_train, x_test, y_train, y_test):
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(x_test.shape[1],)))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    y_pred = model.predict_classes(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_test, y_pred
