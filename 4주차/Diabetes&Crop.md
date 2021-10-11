```py
#Diabetes
W = torch.zeros((8, 1), requires_grad=True, device="cuda") #(8,1)? (8,2)?
b = torch.zeros(1, requires_grad=True, device="cuda") #1? 2?

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.0001)

nb_epochs = 1000000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(train_x.matmul(W) + b)
    cost = -(train_y * torch.log(hypothesis) + (1 - train_y) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100000번마다 로그 출력
    if epoch % 100000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```

```py
hypothesis = torch.sigmoid(test.matmul(W) + b)
predict = hypothesis >= 0.5
```

```py
#Crop
#실수화
from sklearn.preprocessing import LabelEncoder

classle = LabelEncoder()
train['label'] = classle.fit_transform(train['label'])

print(np.unique(train['label']))
```

```py
#Crop
W = torch.zeros((7, 22), requires_grad=True) 
b = torch.zeros(1, requires_grad=True) 

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 100000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = F.softmax(train_x.matmul(W) + b, dim=1)
    cost = F.cross_entropy((train_x.matmul(W) + b), train_y)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 10000번마다 로그 출력
    if epoch % 10000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
```

```py
hypothesis = F.softmax(test.matmul(W) + b, dim=1)
predict = torch.argmax(hypothesis, dim=1)
```

```py
# 제출파일에 저장하는 과정
submit = pd.read_csv('../input/2021-ai-w4-p2/sample.csv')

predict = classle.inverse_transform(predict)

for i in range(len(predict)):
    submit['label'][i]=predict[i]

submit.to_csv('submit.csv', index = False)
submit
```
