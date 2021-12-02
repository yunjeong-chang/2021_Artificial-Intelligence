```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder 

from sklearn.model_selection import train_test_split
#pip install pretrainedmodels
import pretrainedmodels
from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed(777)
```
```py
df_data=pd.read_csv('../input/2021-ai-w11-p1/Label2Names.csv',header=None)
train_root='../input/2021-ai-w11-p1/train_csv_v2'
test_root='../input/2021-ai-w11-p1/test_csv_v2'

class trainDataset(Dataset):
    def __init__(self, train_root):
        self.trainlabel=[]
        self.trainpath=[]
        for classname in tqdm(os.listdir(train_root)):
            ## class name to index
            if classname =="BACKGROUND_Google":
                labelindex=102
            else:
                labelindex=(df_data.index[df_data[1]==classname]+1).tolist()[0]
            
            for csvname in os.listdir(os.path.join(train_root,classname)):
                self.trainlabel.append(labelindex-1)
                ## 데이터 경로 저장 
                csvpath=os.path.join(train_root,classname,csvname)
                self.trainpath.append(csvpath)

    def __getitem__(self, idx):
        # index 에 해당하는 label 값 과 영상 데이터 받아오기 
        csvpath=self.trainpath[idx]
        label=self.trainlabel[idx]
        # 1D 데이터를 영상으로 변환 
        img=np.array(pd.read_csv(csvpath)).reshape((256,256,3)) 
        img=img.transpose((2,0,1))
        # numpy 데이터를 tensor 형태로 변환 
        img=torch.from_numpy(img).float()
        label=torch.tensor(label)
        return img, label
    
    def __len__(self):
        return len(self.trainpath)
    
class testDataset(Dataset):
    def __init__(self, test_root):

        self.testlabel=[]
        self.testpath=[]
        testsort=sorted(os.listdir(test_root))
        for csvname in tqdm(testsort):
            ## 영상 경로 저장 
            csvpath=os.path.join(test_root,csvname)
            self.testpath.append(csvpath)
            self.testlabel.append(csvname)
            
    def __getitem__(self, idx):
        csvpath=self.testpath[idx]
        label=self.testlabel[idx]
        ## 1D 데이터를 영상으로 변환 
        img=np.array(pd.read_csv(csvpath)).reshape((256,256,3))
        img=img.transpose((2,0,1))
        ## numpy 데이터를 tensor 형태로 변환 
        img=torch.from_numpy(img).float()

        return img,label
    
    def __len__(self):
        return len(self.testpath)
    
train_data = trainDataset(train_root)
test_data = testDataset(test_root)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=16)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)
```
```py
class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)
        self.l0 = nn.Linear(512, 102)
        self.dropout = nn.Dropout2d(0.4)
        
    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0
    
model = ResNet34(pretrained=True).to(device)
```
```py
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_correct = 0
    
    for i, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == target).sum().item()
        
        loss.backward()
        optimizer.step()
        
    loss = running_loss/len(train_loader.dataset)
    accuracy = 100. * running_correct/len(train_loader.dataset)
    
    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
```
```py
predictions = np.array([])
with torch.no_grad():
    model.eval()
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        v, pred = torch.max(outputs, 1)
        pred = pred.cpu().numpy()
        predictions = np.concatenate((predictions, pred), axis = None)

submit = pd.read_csv('../input/2021-ai-w11-p1/submission.csv')
submit['Category']= predictions.astype(int) + 1

submit.to_csv('submit.csv', index = False)
submit
```
