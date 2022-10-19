import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def kali():
  print('kali')

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

class cs19b011NN(nn.Module):
  # pass
  # ... your code ...
  # ... write init and forward functions appropriately ...
    def __init__(self):
        super(cs19b011NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
            
        )
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  # model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  model = cs19b011NN().to(device)
  
  print ('Returning model... (rollnumber: 11)', model)
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None
    
  print ('Returning model... (rollnumber: 11)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
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
  print ('Returning metrics... (rollnumber: cs03)')
  
  
  return accuracy_val, precision_val, recall_val, f1score_val
