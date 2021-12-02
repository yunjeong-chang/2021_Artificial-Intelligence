```py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
```
```py
train_x = pd.read_csv('../input/2021-ai-w10-p1/2021-ai-w10-p1/train.csv', dtype = np.float32)
train_y = train_x.pop('Category').astype('int64')

train_x = train_x.to_numpy() / 255.0
train_y = train_y.to_numpy()

train_x = train_x.reshape(-1, 28, 28, 1)
train_y = train_y.reshape(-1,1)
```
```py
class MNISTDataset(Dataset):
    def __init__(self, images, labels): 
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        image = self.transform(image)
        image = image.repeat(3, 1, 1)
        return image, label

    def __len__(self):
        return len(self.images)
    
train_data = MNISTDataset(train_x, train_y)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
```
```py
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.to(device)
```
```py
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

epochs = 20
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, labels.flatten())
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}')
```
```py
test = pd.read_csv('../input/2021-ai-w10-p1/2021-ai-w10-p1/test.csv', dtype=np.float32)
test = test.to_numpy() / 255.0
test = test.reshape(-1, 28, 28, 1)

test_tensor = torch.from_numpy(test).permute(0, 3, 1, 2)
test_tensor = test_tensor.repeat(1, 3, 1, 1)

images= test_tensor.to(device)
outputs = model(images)
_, predictions = torch.max(outputs, 1)
predictions = predictions.cpu()

submit = pd.read_csv('../input/2021-ai-w10-p1/2021-ai-w10-p1/sample_submit.csv')
submit['Category'] = predictions
submit.to_csv("submission.csv", index = False)
submit
```
