import pickle
import keras
import matplotlib.pyplot as plt

from Model.CNN import CNN


# Function for show history
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


# Load data
print("load data......")

X = pickle.load(open("./Data//X_save.save", 'rb'))

y = pickle.load(open("./Data//Y_save.save", 'rb'))


# Define module.
print("Load module......")

model = CNN()
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

model.build((None, 64, 32, 10))

model.summary()

history = model.fit(X.astype(float),
                    y.astype(float),
                    batch_size=30,
                    epochs=50,
                    validation_data=(X.astype(float), y.astype(float)))

show_train_history(history, 'acc', 'val_acc',
                   './Result/History_Acc.jpg')
show_train_history(history, 'loss', 'val_loss',
                   './Result/History_Loss.jpg')

model.save_weights("./ModelSave/CNN.h5")
