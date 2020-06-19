
from sklearn.metrics import f1_score

import torch.nn as nn
import torch.optim as optim
import torch

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F


import numpy as np
import json


print(torch.cuda.is_available())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True

f2 = open("/media/janzz11/Backup_Drive/rs_3_classes.jsonl")

classes = json.load(f2)

number_classes = len(classes)

class CNN_Text(nn.Module):
    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [5, 50, 150, 250, 350]
        num_filters = 24

        #self.convs1 = nn.ModuleList([nn.Conv1d(1, num_filters, (K, 1)) for K in filter_sizes])

        self.dropout = nn.Dropout(0.1)

        #self.fc1 = nn.Linear(len(filter_sizes) * num_filters, len(filter_sizes) * num_filters // 2)

        #self.fc2 = nn.Linear(len(filter_sizes) * num_filters // 2, number_classes)

        self.fc1 = nn.Linear(7044, 3522)

        self.fc2 = nn.Linear(3522, number_classes)

    def forward(self, x):
        #x = x.unsqueeze(1)

        #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        #x = torch.cat(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        logit = F.sigmoid(self.fc2(x))

        return logit

model = CNN_Text()

model.to(device)

loss_function = nn.BCELoss(reduction='mean')

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

dataset = open("/media/janzz11/Backup_Drive/rs_3_fetuared_embedded_flair_bpemb_train.jsonl")

number_samples = 1000000

#x_train = x_train[:c]
#y_train = y_train[:c]

#x_val = torch.tensor(x_train[valid_idx], dtype=torch.float32).to("cpu")
#y_val = torch.tensor(y_train[valid_idx], dtype=torch.float32).to("cpu")

#valid = TensorDataset(x_val, y_val)
#validloader = DataLoader(valid, batch_size=32)


for epoch in range(1, 10):

    train_loss, valid_loss = [], []

    for i in range(1 , 20):

        print("epoch:",epoch,"split:", i)

        c = 0
        c_c = 0

        x_train = np.zeros((number_samples//20, 7044), dtype=np.float32)
        y_train = np.zeros((number_samples//20, number_classes), dtype=np.float32)

        for line in dataset:

            x = json.loads(line)

            c_c+=1

            if c_c<c:
                continue

            try:
                x_train[c,:] = np.array(x["encoded_entity"])

                for elem in x["labels"]:

                    y_train[c,classes[elem]] = 1

                c+=1

            except:

                break

        print("finish -loading data")

        x_train = x_train[:c]
        y_train = y_train[:c]

        x_eval = torch.tensor(x_train[:100], dtype=torch.float32).to("cpu")

        y_eval = y_train[:100]

        #split_size = int(0.9999 * len(x_train))

        split_size = len(x_train)

        index_list = list(range(len(x_train)))

        train_idx, valid_idx = index_list[:split_size], index_list[split_size:]

        ## create iterator objects for train and valid datasets

        x_train = torch.tensor(x_train[train_idx], dtype=torch.float32).to("cpu")
        y_train = torch.tensor(y_train[train_idx], dtype=torch.float32).to("cpu")

        print(x_train.shape)
        print(y_train.shape)

        train = TensorDataset(x_train, y_train)

        trainloader = DataLoader(train, batch_size=32, shuffle=True)

         ## training part

        print("start optimizing split")

        for data, target in trainloader:

            optimizer.zero_grad()

            output = model(data.to(device))

            loss = loss_function(output, target.to(device))

            loss.backward()
            optimizer.step()
            tr_loss = loss.item()

            train_loss.append(tr_loss)

            print("loss:",round(tr_loss,2))
            print()

        y_pred = model(x_eval.to(device)).detach().cpu().numpy()

        f1 = f1_score(y_eval, y_pred, average='macro')

        print("score:",f1)



torch.save(model, "./model_bld_bpe.pth")

