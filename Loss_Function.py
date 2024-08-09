import torch
import torch.nn as nn


def Loss_Function(prediction: torch.tensor, tgt_gt: torch.tensor):
    '''
    Function for compute loss
    '''

    # Loss function : cross entropy
    criterion = nn.CrossEntropyLoss()

    loss = 0.0

    # Compute loss with each example
    for example in range(len(prediction)):

        loss += criterion(prediction[example, :].contiguous().view(-1, 10),
                          tgt_gt[example, :].contiguous().view(-1))

    # Update parameters
    loss /= float(len(prediction))

    return loss
