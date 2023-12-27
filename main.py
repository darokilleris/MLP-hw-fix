from operator import getitem

import numpy as np
from sklearn.neural_network import MLPClassifier
import torch
from datasets.KMNIST import KMNIST
from configs.kmnist_cfg import cfg as kmnist_cfg
from configs.train_cfg import cfg as train_cfg
from configs.mlp_cfg import cfg as model_cfg
from models.MLP import MLP
import torch.nn as nn
import torch.optim


a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
c = []
c.append(a)
c.append(b)
print(torch.cat([a,b]))

print('getting train dataset...')
train_dataset = KMNIST(cfg = kmnist_cfg,dataset_type='train')
print('done!')

print('getting test dataset...')
test_dataset = KMNIST(cfg = kmnist_cfg,dataset_type='test')
print('done!')

model = MLP(model_cfg)
print(model.number_of_parameters())

train_dataset.show_statistics()
test_dataset.show_statistics()


print('!_!')