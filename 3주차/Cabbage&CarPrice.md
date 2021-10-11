```py
#Cabbage
# 모델 초기화
W = torch.zeros((4, 1), requires_grad=True) 
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.001)

nb_epochs = 80000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    # Matrix 연산!!
    hypothesis = train_x.matmul(W) + b 

    # cost 계산
    cost = torch.mean((hypothesis - train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 1000번마다 로그 출력
    if epoch % 1000 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))    
print(W)
print(b)
```

```py
predict = test.matmul(W) + b 
predict
```

```py
#CarPrice
#실수화
from sklearn.preprocessing import LabelEncoder

classle = LabelEncoder()
train['model'] = classle.fit_transform(train['model'].values)
test['model'] = classle.fit_transform(test['model'].values)
```

```py
#CarPrice
# 모델 초기화
W = torch.zeros((9, 1), requires_grad=True, device="cuda")
b = torch.zeros(1, requires_grad=True, device="cuda")

#loss 설정 (MAE)
loss = torch.nn.L1Loss()
# optimizer 설정
optimizer = optim.Adam([W, b], lr=0.001)

nb_epochs = 10000000
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    # Matrix 연산!!
    hypothesis = train_x.matmul(W) + b 

    # cost 계산 
    cost = loss(hypothesis, train_y)
    #cost = torch.mean((hypothesis - train_y) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 1000000번마다 로그 출력
    if epoch % 1000000 == 0 :
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))    
print(W)
print(b)
```
```py
# 학습을 통해 알아낸 W와 b를 이용해 예측값 구하기
with torch.no_grad():
    predict = test.matmul(W) + b 
predict

# 제출파일에 저장하는 과정
submit = pd.read_csv('../input/2021-ai-w3-p2/sample_submit.csv')

for i in range(len(predict)):
    submit['price'][i]=predict[i].item()

submit.to_csv('submit.csv', index = False)
submit
```
