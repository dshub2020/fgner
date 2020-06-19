import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.optim as optim
import torch
import spacy
import random


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True


class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [5, 50, 150, 250, 300, 500]
        num_filters = 24

        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, 1)) for K in filter_sizes])

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, len(filter_sizes) * num_filters // 2)

        self.fc2 = nn.Linear(len(filter_sizes) * num_filters // 2, 10)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        logit = F.sigmoid(self.fc2(x))

        return logit

model = CNN_Text()

model.to(device)

loss_function = nn.BCEWithLogitsLoss(reduction='mean')

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)


for epoch in range(1, 10):

    train_loss, valid_loss = [], []

    for i in range(1,5):

        print("epoch:",epoch,"split:", i)

        c = 0

        x_train = np.zeros((4301908, ))
        y_train = np.zeros((4301908, 300))

        dataset = open("./preprocessing/data_all"+str(i)+".json","r",encoding='utf-8')

        for line in dataset:

            x = json.loads(line)

            try:
                x_train[c,:] = np.array(x["input"])
                y_train[c,:] = np.array(x["target"])

                #print(round( (c / (3301908*4)) * 100, 2), "%")
                c+=1

            except:

                break

        print("finish -loading data")

        x_train = x_train[:c]
        y_train = y_train[:c]

        #split_size = int(0.9999 * len(x_train))

        split_size = len(x_train)

        index_list = list(range(len(x_train)))

        train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

        ## create iterator objects for train and valid datasets

        x_val = torch.tensor(x_train[valid_idx], dtype=torch.long).to("cpu")
        y_val = torch.tensor(y_train[valid_idx], dtype=torch.float32).to("cpu")

        x_train = torch.tensor(x_train[train_idx], dtype=torch.long).to("cpu")
        y_train = torch.tensor(y_train[train_idx], dtype=torch.float32).to("cpu")

        #print(x_train.shape)
        #print(y_train.shape)

        train = TensorDataset(x_train, y_train)

        trainloader = DataLoader(train, batch_size=64, shuffle=True)

        #valid = TensorDataset(x_val, y_val)
        #validloader = DataLoader(valid, batch_size=32)

         ## training part

        print("start optimizing split")

        for data, target in trainloader:

            optimizer.zero_grad()
            output = model(data.to(device))

            loss = loss_function(output, target.to(device), torch.tensor(1,dtype=torch.float32).to(device))

            loss.backward()
            optimizer.step()
            tr_loss = loss.item()
            train_loss.append(tr_loss)

        print("loss:",round(tr_loss,2))
        print()


torch.save(model, "./model_bld_bpe.pth")

