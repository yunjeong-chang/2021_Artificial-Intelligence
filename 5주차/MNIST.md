```py
W = torch.zeros((784, 10), requires_grad=True, device="cuda") 
b = torch.zeros(1, requires_grad=True, device="cuda") 

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.00001) #LR도 그렇고

nb_epochs = 100000 #EPOCH도 그렇고 작아지고, 커질수록 성능은 좋아지는데 언제까지..? 배치는 아직 안건드려봄!
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
submit = pd.read_csv('../input/2021-ai-w5-p1/sample_submit.csv')

for i in range(len(predict)):
    submit['label'][i]=predict[i].item()

submit=submit.astype(np.int32)
submit.to_csv('submit.csv', index = False)
submit
```
