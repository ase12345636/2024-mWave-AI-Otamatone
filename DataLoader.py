import h5py
from einops import rearrange
import numpy as np
import torch
from Config import device


def DataLoader():
    f = h5py.File('TestData.h5', 'r')

    dataset = f['DS1'][:]
    dataset = np.array(dataset, dtype=np.float32)
    dataset = rearrange(dataset, '(b c) h w f ->b f c h w', b=1)

    label = f['LABEL'][-1, :]
    label = np.array(label, dtype=np.float32)
    label = rearrange(label, '(b c) -> b c', b=1)

    return torch.from_numpy(dataset).to(torch.float32).to(device), torch.from_numpy(label).to(torch.long).to(device)
