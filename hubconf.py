import torch
from torch import nn

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname


class cs19b011NN(nn.Module):
    # ... your code ...
    # ... write init and forward functions appropriately ...
    def __init__(self, in_channels, w, h, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=10, kernel_size=(2, 2), padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=(2, 2), padding='same')
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
    model = None

    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # Use softmax and cross entropy loss functions
    # set model variable to proper object, make use of train_data

    # In addition,
    # Refer to config dict, where learning rate is given,
    # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
    # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
    # You need to create 2d convoution layers as per specification above in each element
    # You need to add a proper fully connected layer as the last layer

    # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
    # HINT: Flatten function can also be used if required
    return model

    print('Returning model... (rollnumber: xx)')

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
