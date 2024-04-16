
import torch.nn as nn
import torch.nn.functional as F
import torch



class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        # write your codes here
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, img):
        # write your codes here
        x = self.conv1(img)
        x = F.tanh(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.tanh(x)
        x = F.max_pool2d(x, kernel_size=2)
        

        x = x.view(-1, 16 * 4 * 4)  # reshape to fit fully connected layer
        # apply dropout
        x = self.drop_layer(x)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        output = self.fc3(x)

        output = F.softmax(output,dim=1)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=56)
        self.fc2 = nn.Linear(in_features=56, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=10)

    def forward(self, img):
        x = torch.flatten(img, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        output = F.softmax(x,dim=1)

        return output
