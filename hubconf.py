import torch
from torch import nn

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname


class cs19b011NN(nn.Module):
    # ... your code ...
    # ... write init and forward functions appropriately ...
    def __init__(self, in_channels, w, h, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=10, 
            kernel_size=(2, 2), 
            padding='same')
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            in_channels=10, 
            out_channels=20, 
            kernel_size=(2, 2), 
            padding='same')
        self.relu2 = nn.ReLU()
        
        # self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=w*h*20, out_features=num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x

# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model(train_data_loader=None, n_epochs=10):
  model = cs19b011NN()  
  
  print ('Returning model... (rollnumber: 11)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
  iter = 0
  

  for epoch in range(n_epochs):

    model = get_model(config["in_channels"],config["out_channels"],config["kernel_size"],config["stride"],config["padding"])
    criterion = nn.CrossEntropyLoss()
    learning_rate = lr
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i, (images, labels) in enumerate(train_data_loader):
        # Load images as tensors with gradient accumulation abilities
        images = images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

  
  iter+=1

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

    print('Returning metrics... (rollnumber: xx)')

    return accuracy_val, precision_val, recall_val, f1score_val
