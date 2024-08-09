import os
import random
import matplotlib.pyplot as plt
import numpy as np

import torch

from pathlib import Path

from torch.cuda.amp import autocast as autocast

from Config import device, lr, betas, eps, num_epoch, batch_multiplier
from torchsummary import summary
from ViViT import ViViT
from Model_Ema import Model_Ema
from DataLoader import DataLoader
from Optimizer import Optimizer
from Loss_Function import Loss_Function


# Load model
model = ViViT().to(device)
# model.load_state_dict(torch.load(Path(PATH+mode+"Model_Best.h5")))
summary(model)

# Load ema model
model_ema = Model_Ema(model)

# Load optimizer
optimizer, sechdualer = Optimizer(model, lr, betas, eps)

# # Load performance metrics
# Acc = Performance_Metrics()

# Records
iteration = 0
count = batch_multiplier
last_loss = float("inf")
min_loss = float("inf")
total_loss = 0.0
iteration_history = []
loss_history = []
lr_history = []

# Load Data
video, ground_truth = DataLoader()
print(video.shape)
print(ground_truth.shape)

# Start training
model.train()

# For each epoch
print("Training...")

for epoch in range(num_epoch):

    optimizer.zero_grad()

    prediction = model(video)

    loss = Loss_Function(prediction=prediction,
                         tgt_gt=ground_truth)
    total_loss += float(loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    model_ema.update(model)

    # Print loss for each iteration
    iteration += 1
    iteration_history.append(iteration)
    loss_history.append(total_loss)
    lr_history.append(sechdualer.get_lr())
    print(f"Iteration: {iteration}, Loss: {total_loss}")
    print(sechdualer.get_lr())

    # Update recodes
    count = batch_multiplier
    last_loss = total_loss
    min_loss = min(min_loss, last_loss)
    total_loss = 0.0

    # print("Save...")

    # # Save best model
    # if min_loss == last_loss or iteration == 1:
    #     torch.save(model.state_dict(), Path(
    #         PATH+mode+"Model_Best.h5"))
    #     torch.save(model_ema.model.state_dict(), Path(
    #         PATH+mode+"Model_Ema_Best.h5"))

    # # Save last model
    # torch.save(model.state_dict(), Path(
    #     PATH+mode+"Model_Last.h5"))
    # torch.save(model_ema.model.state_dict(), Path(
    #     PATH+mode+"Model_Ema_Last.h5"))

    sechdualer.step()

print(f"Finish training\nMin_loss = {min_loss}\nLast_loss = {last_loss}")

plt.plot(np.array(iteration_history),
         np.array(loss_history), label='training loss')
plt.savefig(Path("loss_history.png"))
plt.show()

plt.plot(np.array(iteration_history),
         np.array(lr_history), label='training loss')
plt.savefig(Path("lr_history.png"))
plt.show()
