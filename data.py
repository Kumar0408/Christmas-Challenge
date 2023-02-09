from torch.utils.data import Dataset
import torch
import torchvision
import os
from PIL import Image
import natsort
#from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models, transforms


class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        self.training = training
        self.path = path
        
        #For training data
        self.train_transform = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(degrees=10),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        #For validation data
        self.val_transform = transforms.Compose([
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean= ([0.485, 0.456, 0.406]),std = ([0.229, 0.224, 0.225]))
             ])
         
        if self.training == True:
            self.dataset = ImageFolder(path + './train',transform=self.train_transform)
        else:
            self.path = path
            self.sorted_image = natsort.natsorted(os.listdir(self.path))
    
    def __len__(self):
        return len(self.dataset)
            
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training == True:
            img,label = self.dataset[index]
            return img,label
        
        else:
            img_loc = os.path.join(self.path,self.sorted_image[index])
            
            img = Image.open(img_loc).convert("RGB")
            img = self.val_transform(img)
            return (img, )
        raise NotImplementedError
