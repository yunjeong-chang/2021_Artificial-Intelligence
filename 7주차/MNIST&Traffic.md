```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
```

```py
#데이터 불러오기
train = pd.read_csv('../input/2021-ai-w7-p1/mnist_train.csv')
test = pd.read_csv('../input/2021-ai-w7-p1/mnist_test.csv')
```

```py
train_x = train.iloc[:,1:-1]
train_y = train.iloc[:,-1]
test = test.iloc[:,1:]
```

```py
train_x = np.array(train_x)
train_y = np.array(train_y)
test = np.array(test)

from sklearn import preprocessing
Scaler = preprocessing.StandardScaler()
train_x = Scaler.fit_transform(train_x)
test = Scaler.transform(test)

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.LongTensor(train_y).to(device) 
test = torch.FloatTensor(test).to(device)

print(train_x.shape)
print(train_y.shape)
print(test.shape)
```

```py
train_dataset = torch.utils.data.TensorDataset(train_x,train_y)

data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = 30,
                                          shuffle = True,
                                          drop_last=True)
```

```py
linear1 = torch.nn.Linear(14,128,bias=True)
linear2 = torch.nn.Linear(128,256,bias=True)
linear3 = torch.nn.Linear(256,128,bias=True)
linear4 = torch.nn.Linear(128,64,bias=True)
linear5 = torch.nn.Linear(64,1,bias=True)
relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.3)

torch.nn.init.xavier_uniform_(linear1.weight) #xavier_normal_
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
```

```py
model = torch.nn.Sequential(linear1,relu,dropout,
                            linear2,relu,dropout,
                            linear3,relu,dropout,
                            linear4,relu,dropout,
                            linear5).to(device)
```

```py
# 손실함수와 최적화 함수
loss = torch.nn.CrossEntropyLoss().to(device)
#loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

```py
total_batch = len(data_loader)
model.train()

for epoch in range(100+1): #(training_epoch)
    avg_cost = 0
    for X,Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = loss(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch
        
    if epoch % 10 == 0:
        print('Epoch {:4d} / Cost: {:.6f}'.format(epoch, avg_cost))
```

```py
with torch.no_grad(): 
    model.eval()
    prediction = model(test)
    prediction = torch.argmax(prediction, 1)
prediction
```

```py
# 제출파일에 저장하는 과정
submit = pd.read_csv('../input/2021-ai-w7-p1/submission.csv')

prediction = prediction.cpu().numpy().reshape(-1,1)
submit['Label']=prediction

#submit=submit.astype(np.int32)
submit.to_csv('submit.csv', index = False)
submit
```
