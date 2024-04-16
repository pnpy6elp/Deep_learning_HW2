import os
import torch
from torchvision import transforms
from skimage import io
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.image as mpimg
# import some packages you need here


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        # write your codes here
        self.data_dir = data_dir
        # apply data augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Extract labels from filenames
        self.labels = {}
        for filename in os.listdir(data_dir):
            label = int(filename.split('_')[-1].split('.')[0])
            idx = int(filename.split('_')[0])
            self.labels[idx] = label
        

    def __len__(self):
        # write your codes here
        return len(self.labels)

    def __getitem__(self, idx):
        # write your codes here
        img_name = os.listdir(self.data_dir)[idx]
        idx2 = int(img_name.split('_')[0])
        img_path = os.path.join(self.data_dir, img_name)
        # image = io.imread(img_path)
        #image =  Image.open(img_path)
        image =  mpimg.imread(img_path)
        img = self.transform(image)
        label = self.labels[idx2]

        return img, label
    
    
class MNIST2(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        # write your codes here
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Extract labels from filenames
        self.labels = {}
        for filename in os.listdir(data_dir):
            label = int(filename.split('_')[-1].split('.')[0])
            idx = int(filename.split('_')[0])
            self.labels[idx] = label
        

    def __len__(self):
        # write your codes here
        return len(self.labels)

    def __getitem__(self, idx):
        # write your codes here
        img_name = os.listdir(self.data_dir)[idx]
        idx2 = int(img_name.split('_')[0])
        img_path = os.path.join(self.data_dir, img_name)
        # image = io.imread(img_path)
        #image =  Image.open(img_path)
        image =  mpimg.imread(img_path)
        img = self.transform(image)
        label = self.labels[idx2]

        return img, label

if __name__ == '__main__':
    # write test codes to verify your implementations

    trainset = MNIST('../data/train')
    testset = MNIST('../data/test')
    
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    print("# of trainset : ",len(trainset))
    print("# of testset : ",len(testset))

    # Output >>
    # of trainset :  60000
    # of testset :  10000

    sample_idx = [0,5,10,4020]
    for i in sample_idx:
        img, label = trainset[i]
        print(f"Train : {img.shape} shape, {label} label")

        img, label = testset[i]
        print(f"Test : {img.shape} shape, {label} label")

    # Output >> 
    # Train : torch.Size([1, 28, 28]) shape, 8 label
    # Test : torch.Size([1, 28, 28]) shape, 1 label
    # Train : torch.Size([1, 28, 28]) shape, 7 label
    # Test : torch.Size([1, 28, 28]) shape, 7 label
    # Train : torch.Size([1, 28, 28]) shape, 5 label
    # Test : torch.Size([1, 28, 28]) shape, 6 label
    # Train : torch.Size([1, 28, 28]) shape, 3 label
    # Test : torch.Size([1, 28, 28]) shape, 4 label

