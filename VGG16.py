from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam


def VGG16():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3), padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding='same'))
    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding='same'))
    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(3, activation="relu"))

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
