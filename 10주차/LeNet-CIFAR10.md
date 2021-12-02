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
from glob import glob
from os.path import join
from PIL import Image
from tqdm import tqdm

class CIFAR10_DateLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, split, transform=None):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.transform = transform
        self.data = data_path
        if self.split == "TRAIN":
            self.label = [int(p.split('/')[-2]) for p in data_path]
        self.data_len = len(self.data)
            
    def __len__(self):
        return self.data_len 

    def __getitem__(self, index):
        image = Image.open(self.data[index], mode='r')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.split == "TEST":
            return image
        elif self.split == "TRAIN":
            self.label[index] = np.array(self.label[index])
            return image, torch.from_numpy(self.label[index])

    
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_image_path = glob(join('../input/2021-ai-w10-p2/images/images/', '*', '*'))
test_image_path = glob(join('../input/2021-ai-w10-p2/test_data/test_data', '*'))

test_paths=[]
for i in range(len(test_image_path)):
    test_paths.append('../input/2021-ai-w10-p2/test_data/test_data/'+str(i)+'.png')

train_data = CIFAR10_DateLoader(train_image_path, 'train', transform=transform)
test_data = CIFAR10_DateLoader(test_paths, 'test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
```
```py
method = 'GoogLeNet'

if method is "ResNet":
    model = models.resnet18(pretrained=True)
    model.to(device)
    model.fc.out_features = 10                
if method is "VGG":
    model = models.vgg16(pretrained=True)
    model.to(device)
    model.classifier[6].out_features = 10

if method is "GoogLeNet":
    model = models.googlenet(pretrained=True)
    model.to(device)
    model.fc.out_features = 10
```
```py
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
model.train()
for epoch in tqdm(range(epochs)):
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
def test(model, test_dataloader):
    model.eval()
    preds=[]
    for i, data in tqdm(enumerate(test_dataloader)):
        data = data.to(device)
        output = model(data)
        _, pred = torch.max(output.data, 1)
        preds.extend(pred.detach().cpu().tolist())
    
    return preds

prediction = test(model, test_loader)

submit = pd.read_csv('../input/2021-ai-w10-p2/format.csv')
submit['label'] = prediction
submit.to_csv("submission.csv", index = False)
submit
```
