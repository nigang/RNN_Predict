from dataloader import *
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data

DATA_BASE = r'C:\Users\gni\Desktop\tutorial\LotteryRecode.sqlite'
TABLE_NAME = r'lottery'

INPUT_SIZE = 3
TIME_WINDOW = 10
RNN_LAYER = 3
HIDDEN_SIZE = 5

EPOCH = 10
DATA_SIZE = 1000
TEST_SIZE = 50


def load_data_from_db():
    loader = LotteryRecordLoad(DATA_BASE)
    item_list = loader(TABLE_NAME, LotteryRecord, order_by='date', reverse=True, num=DATA_SIZE)
    item_list.sort(key=lambda ele: ele.date)
    items = [[int(item.hundred), int(item.decade), int(item.unit)] for item in item_list]

    train_data = list()
    for i in range(len(items) - TIME_WINDOW):
        in_data = items[i:i+TIME_WINDOW]
        out_data = sum(items[i+TIME_WINDOW])
        train_data.append((in_data, [out_data]))
    return train_data[:-TEST_SIZE], train_data[-TEST_SIZE:]



TRAIN_DATA, TEST_DATA = load_data_from_db()


class LotteryDataset(Dataset):
    def __init__(self, data):
        super(LotteryDataset, self).__init__()
        self.data_set = data

    def __getitem__(self, index):
        return torch.Tensor(self.data_set[index][0])/9, torch.Tensor(self.data_set[index][1])/27

    def __len__(self):
        return len(self.data_set)


class LSTMPred(nn.Module):
    def __init__(self):
        super(LSTMPred, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=RNN_LAYER,
            batch_first=True,
        )
        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


train_loader = DataLoader(dataset=LotteryDataset(TRAIN_DATA), batch_size=32, shuffle=False)
test_x = torch.Tensor([ele[0] for ele in TEST_DATA])/9
test_y = torch.Tensor([ele[1] for ele in TEST_DATA])/27
model = LSTMPred()
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, TIME_WINDOW, INPUT_SIZE))
        b_y = Variable(y)
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            predict_y = model(test_x)
            import pdb
            pdb.set_trace()




