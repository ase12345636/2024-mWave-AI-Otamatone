import os
import h5py
from einops import rearrange
import numpy as np
import pickle


dirs = os.listdir("./Data//Otamatone//")

frame_total = 190
window_size = 12
hop_length = 1

cnt = 0
for file in dirs:
    cnt += 1

    print("Split frame")
    X = np.array(None)

    f = h5py.File("./Data//Otamatone//"+file, 'r')

    images = f['DS1'][:]
    images = np.array(images, dtype=np.float32)
    images = rearrange(images, 'c h w f ->f c h w')

    for i in range(0, frame_total-window_size+1, hop_length):
        tmp = rearrange(images[i:i+window_size],
                        '(d f) c h w -> d (c h) w f', d=1)

        if (X == np.array(None)).all():
            X = tmp

        else:
            X = np.append(X, tmp, axis=0)

    print(X.shape)
    pickle.dump(
        X, open("./Data//PropreccessedData//X//X_save_"+str(cnt)+".save", 'wb'))

    print("Split label")
    Y = np.array(None)

    label = f['LABEL'][:]
    label = np.array(label, dtype=np.float32)
    label = rearrange(label, 'f l -> (f l)')

    label_cnt = 0
    for i in range(window_size-1, frame_total, hop_length):
        tmp = np.array(0)
        if label[i-window_size+1:i+1].all():
            label_cnt += 1
            tmp = np.array(label_cnt)

        if (Y == np.array(None)).all():
            Y = tmp

        else:
            Y = np.append(Y, tmp)

    print(Y.shape)
    pickle.dump(
        Y, open("./Data//PropreccessedData//Y//Y_save_"+str(cnt)+".save", 'wb'))

    f.close
