import os
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from Model.CNN import CNN


# Function for showing history
def show_train_history(history, train_type, test_type, outputfilename):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history[train_type])
    plt.plot(history.history[test_type])
    plt.title('Train History')
    if train_type == 'acc':
        plt.ylabel('Accuracy')
    else:
        plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig(outputfilename)
    plt.close(fig)


# Define module.
print("Load Module......")

model = CNN()

model.build((None, 64, 32, 12))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])


# Load data
print("Get Data......")

X = np.array(None)
y = np.array(None)

dirs_X = os.listdir("./Data//PropreccessedData//X//")
dirs_X.sort()

dirs_Y = os.listdir("./Data//PropreccessedData//Y//")
dirs_Y.sort()


# Training
print("Start Training......")

epoch = 50
file_length = len(dirs_X)

for i in range(epoch):
    for part in range(file_length):
        X = pickle.load(
            open("./Data//PropreccessedData//X//"+dirs_X[part], 'rb'))

        y = pickle.load(
            open("./Data//PropreccessedData//Y//"+dirs_Y[part], 'rb'))

        print("Epoch : "+str(i+1))
        history = model.fit(X.astype(float),
                            y.astype(float),
                            batch_size=90,
                            epochs=1,
                            validation_data=(X.astype(float), y.astype(float)))

model.save_weights("./ModelSave/CNN.h5")


show_train_history(history, 'acc', 'val_acc',
                   './Result/History_Acc.jpg')
show_train_history(history, 'loss', 'val_loss',
                   './Result/History_Loss.jpg')
