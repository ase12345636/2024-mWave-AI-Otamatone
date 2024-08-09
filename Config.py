import torch


device = torch.device('cuda')


training_max_num_batch = 1
training_max_num_batch_size = 30

validation_max_num_batch = 1
validation_max_num_batch_size = 30

test_max_num_batch = 1
test_max_num_batch_size = 30

dir = [["train", training_max_num_batch, training_max_num_batch_size],
       ["validation", validation_max_num_batch, validation_max_num_batch_size],
       ["test", test_max_num_batch, test_max_num_batch_size]]


batch_multiplier = 1
warm_up_epoch = 10
num_epoch = 300


lr = 1e-4
betas = (0.9, 0.95)
eps = 1e-9

guidance_scale = 0.2

PATH = ".//ModelSave//"
