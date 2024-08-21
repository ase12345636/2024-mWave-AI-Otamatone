from keras import layers
import keras


class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(input_shape=[64, 32, 12], filters=64, kernel_size=(7, 7),
                                   activation="relu", padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")

        self.conv2 = layers.Conv2D(filters=64, kernel_size=(7, 7),
                                   activation="relu", padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), padding="same")

        self.conv3 = layers.Conv2D(filters=128, kernel_size=(5, 5),
                                   activation="relu", padding="same")
        self.pool3 = layers.MaxPooling2D(
            pool_size=(2, 2), padding="same")

        self.flat = layers.Flatten()

        self.dense1 = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.5)

        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flat(x)

        x = self.dense1(x)
        x = self.dropout(x)

        predict = self.dense2(x)

        return predict
