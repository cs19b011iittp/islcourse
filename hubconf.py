import torch
from torch import nn

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class cs19b037NN(nn.Module):
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


class cs19b037AdvancedNN(nn.Module):
    def __init__(self, modules_list, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linears = nn.ModuleList(modules_list)
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(in_features=w*h*20, out_features=num_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
        x = self.flatten(x)
        x = nn.Linear(in_features=x.shape[-1],
                      out_features=self.num_classes)(x)
        x = self.softmax(x)
        return x

# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model(train_data_loader=None, n_epochs=10):

    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # Use softmax and cross entropy loss functions
    # set model variable to proper object, make use of train_data

    # print(train_data_loader.dataset)
    # print(train_data_loader)
    # print(len(train_data_loader.dataset))

    w = 0
    h = 0
    num_channels = 0
    num_classes = 0
    for (X, y) in train_data_loader:
        # print(X.shape)
        # print(y.shape)
        num_channels = X.shape[1]
        w = X.shape[2]
        h = X.shape[3]
        num_classes = torch.max(y).item()-torch.min(y).item()+1
        break

    model1 = cs19b037NN(num_channels, w, h, num_classes).to(device)

    optimizer = torch.optim.SGD(model1.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # training
    for epoch in range(n_epochs):
        model1.train()
        train_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(train_data_loader):
            X, y = X.to(device), y.to(device)
            ypred = model1(X)
            loss = loss_fn(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            correct += (ypred.argmax(1) == y).type(torch.float).sum().item()
        print("Epoch", epoch, "accuracy:",
              correct/len(train_data_loader.dataset))

    print('Returning model... (rollnumber: cs19b037)')

    return model1

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):

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

    num_classes = 0
    for (X, y) in train_data_loader:
        num_classes = torch.max(y).item()-torch.min(y).item()+1
        break

    modules_list = []
    if(config is not None):
        for module in config:
            modules_list.append(nn.Conv2d(
                in_channels=module[0], out_channels=module[1], kernel_size=module[2], stride=module[3], padding=module[4]))

    modules_list=modules_list.to(device)
    model1 = cs19b037AdvancedNN(
        modules_list=modules_list, num_classes=num_classes).to(device)

    optimizer = torch.optim.SGD(model1.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # training
    for epoch in range(n_epochs):
        model1.train()
        train_loss = 0
        correct = 0
        for batch, (X, y) in enumerate(train_data_loader):
            X, y = X.to(device), y.to(device)
            ypred = model1(X)
            loss = loss_fn(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss
            correct += (ypred.argmax(1) == y).type(torch.float).sum().item()
        print("Epoch", epoch, "accuracy:",
              correct/len(train_data_loader.dataset))

    print('Returning model... (rollnumber: cs19b037)')

    return model1

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
