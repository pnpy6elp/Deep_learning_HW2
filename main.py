
import dataset
from model import LeNet5, CustomMLP

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """

    # write your codes here

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(trn_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        #correct += torch.sum(predicted == labels.data)
        correct += predicted.eq(labels).sum().item()

    trn_loss = running_loss / len(trn_loader)
    acc = correct / len(trn_loader.dataset)
    acc *= 100 



    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(tst_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    tst_loss = running_loss / len(tst_loader)
    acc = 100. * correct / total

    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here

        # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Hyperparameters
    lr = 0.01
    momentum = 0.9
    num_epochs = 20


    trainset = dataset.MNIST('../data/train') # apply data augmentation
    testset = dataset.MNIST2('../data/test') # not apply data augementation

    # Data loaders
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=8)

    # Model initialization
    model_lenet5 = LeNet5().to(device)
    model_custom_mlp = CustomMLP().to(device)

    # Optimization and loss function
    criterion = nn.CrossEntropyLoss().to(device)
    # apply L2 regularization
    optimizer_lenet5 = optim.SGD(model_lenet5.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)
    optimizer_custom_mlp = optim.SGD(model_custom_mlp.parameters(), lr=lr, momentum=momentum, weight_decay=1e-5)

    # Lists to store statistics for plotting
    lenet5_train_loss_history = []
    lenet5_train_acc_history = []
    lenet5_test_loss_history = []
    lenet5_test_acc_history = []

    custom_mlp_train_loss_history = []
    custom_mlp_train_acc_history = []
    custom_mlp_test_loss_history = []
    custom_mlp_test_acc_history = []

    # Training and testing loop
    for epoch in range(num_epochs):
        # Train LeNet-5
        trn_loss, trn_acc = train(model_lenet5, train_loader, device, criterion, optimizer_lenet5)
        tst_loss, tst_acc = test(model_lenet5, test_loader, device, criterion)
        lenet5_train_loss_history.append(trn_loss)
        lenet5_train_acc_history.append(trn_acc)
        lenet5_test_loss_history.append(tst_loss)
        lenet5_test_acc_history.append(tst_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], LeNet-5 Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%')

    

    for epoch in range(num_epochs):
        

        # Train Custom MLP
        trn_loss, trn_acc = train(model_custom_mlp, train_loader, device, criterion, optimizer_custom_mlp)
        tst_loss, tst_acc = test(model_custom_mlp, test_loader, device, criterion)
        custom_mlp_train_loss_history.append(trn_loss)
        custom_mlp_train_acc_history.append(trn_acc)
        custom_mlp_test_loss_history.append(tst_loss)
        custom_mlp_test_acc_history.append(tst_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Custom MLP Train Loss: {trn_loss:.4f}, Train Acc: {trn_acc:.2f}%, Test Loss: {tst_loss:.4f}, Test Acc: {tst_acc:.2f}%')

    plt.figure(figsize=(12, 4))
    plt.plot(lenet5_train_loss_history, label='Train Loss')
    plt.plot(lenet5_test_loss_history, label='Test Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(lenet5_train_acc_history, label='Train Acc')
    plt.plot(lenet5_test_acc_history, label='Train Acc')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(custom_mlp_train_loss_history, label='MLP Train Loss')
    plt.plot(custom_mlp_test_loss_history, label='MLP Test Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(custom_mlp_train_acc_history, label='MLP Train Acc')
    plt.plot(custom_mlp_test_acc_history, label='MLP Test Acc')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(lenet5_train_loss_history, label='LeNet-5 Train Loss')
    plt.plot(lenet5_test_loss_history, label='LeNet-5 Test Loss')
    plt.plot(custom_mlp_train_loss_history, label='MLP Train Loss')
    plt.plot(custom_mlp_test_loss_history, label='MLP Test Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12, 4))
    plt.plot(lenet5_train_acc_history, label='LeNet-5 Train Acc')
    plt.plot(lenet5_test_acc_history, label='LeNet-5 Test Acc')
    plt.plot(custom_mlp_train_acc_history, label='MLP Train Acc')
    plt.plot(custom_mlp_test_acc_history, label='MLP Test Acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
