import h5py
from einops import rearrange
import numpy as np
import pickle


# def DataLoader():
#     f = h5py.File('TestData.h5', 'r')data

#     data = f['DS1'][:]
#     data = np.array(data, dtype=np.float32)
#     data = rearrange(data, '(b c) h w f ->b f c h w', b=1)

#     label_tmp = f['label_tmp'][-1, :]
#     label_tmp = np.array(label_tmp, dtype=np.float32)
#     label_tmp = rearrange(label_tmp, '(b c) -> b c', b=1)

#     return


f = h5py.File('./Data//TestData.h5', 'r')

images = f['DS1'][:]
images = np.array(images, dtype=np.float32)
images = rearrange(images, 'c h w f ->f c h w')
print(images.shape)

frame_total = len(images)
window_size = 10
hop_length = 1

X = np.array(None)
for i in range(0, frame_total-window_size+1, hop_length):
    tmp = rearrange(images[i:i+window_size], '(d f) c h w -> d (c h) w f', d=1)

    if (X == np.array(None)).all():
        X = tmp

    else:
        X = np.append(X, tmp, axis=0)

    print(X.shape)
pickle.dump(X, open("./Data//X_save.save", 'wb'))


label = f['LABEL'][:]
label = np.array(label, dtype=np.float32)
label = rearrange(label, 'f l -> (f l)')
print(label.shape)

Y = np.array(None)
for i in range(window_size-1, frame_total, hop_length):
    tmp = label[i]

    if (Y == np.array(None)).all():
        Y = tmp

    else:
        Y = np.append(Y, tmp)

    print(Y.shape)
pickle.dump(Y, open("./Data//Y_save.save", 'wb'))
