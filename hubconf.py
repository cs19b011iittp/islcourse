# Debeshee

dependencies = ['torch']

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():

    train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
    )
    test_data = datasets.FashionMNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )

    return train_data, test_data
    
def get_dataloaders(train_data, test_data):
    loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    return loaders


class cs19b011NN(nn.Module):
    def __init__(self):
        super(cs19b011NN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
    model = cs19b011NN()
    loss_func = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   
    return model

    

def train(cnn, loss_func, optimizer, loaders, num_epochs):
    # num_epochs = 10
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        
        pass
    
    
    pass
    PATH = './saved_models/FMNIST_model.pth'
    torch.save(cnn.state_dict(), PATH)


def test_model(model1=None, test_data_loader=None):
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  print(test_data_loader)
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  model1.eval()
  test_loss, correct = 0, 0
  
  with torch.no_grad():
      for X, y in test_data_loader:
          X, y = X.to(device), y.to(device)
          pred = model1(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  print ('Returning metrics... (rollnumber: cs19b011)')
    
  return accuracy_val, precision_val, recall_val, f1score_val
