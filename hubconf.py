import torch
from torch import nn

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname


class cs19b011NN(nn.Module):
    def __init__(cs19b011NN, self, input, output, kernel, strid, pad):

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=self.input, out_channels=self.output, kernel_size=self.kernel, stride=self.strid, padding=self.pad)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10) 

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out

# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model(train_data_loader=None, n_epochs=10):
  model = cs19b011NN()  
  
  print ('Returning model... (rollnumber: 11)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
  iter = 0
  

  for epoch in range(n_epochs):

    model = get_model()
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
