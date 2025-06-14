# ============================================================================
# File: dataset.py
# Date: 2025-03-11
# Author: TA
# Description: Dataset and DataLoader.
# ============================================================================

import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

def get_dataloader(
        dataset_dir,
        batch_size: int = 1,
        split: str = 'test'):
    '''
    Build a dataloader for given dataset and batch size.
    - Args:
        - dataset_dir: str, path to the dataset directory
        - batch_size: int, batch size for dataloader
        - split: str, 'train', 'val', or 'test'
    - Returns:
        - dataloader: torch.utils.data.DataLoader
    '''
    ###############################
    # TODO:                       #
    # Define your own transforms. #
    ###############################
    if split == 'train':
        transform = transforms.Compose([
            # resizes all input images to 32x32 pixels, 
            # ensuring that the input size matches the expected size of the CIFAR-10 images.
            transforms.Resize((32,32)),
            ##### TODO: Data Augmentation Begin #####
            
            # randomly flips the image horizontally with a 50% probability
            transforms.RandomHorizontalFlip(p=0.5),
            # crops a 32x32 region from the image randomly, and pads the image by 4 pixels before cropping
            transforms.RandomCrop(32, padding=4),
            
            ##### TODO: Data Augmentation End #####
            
            # Converts the image into a PyTorch tensor
            transforms.ToTensor(),
            # Normalizes the image by subtracting the mean and dividing by the standard deviation
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    else: # 'val' or 'test'
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            # we usually don't apply data augmentation on test or val data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = CIFAR10Dataset(dataset_dir, split=split, transform=transform)
    if dataset[0] is None:
        raise NotImplementedError('No data found, check dataset.py and implement __getitem__() in CIFAR10Dataset class!')
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(split=='train'),
                            num_workers=0,
                            pin_memory=True, 
                            drop_last=(split=='train'))

    return dataloader

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset_dir, split='test', transform=None):
        super(CIFAR10Dataset).__init__()
        
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        with open(os.path.join(self.dataset_dir, 'annotations.json'), 'r') as f:
            json_data = json.load(f)
        
        self.image_names = json_data['filenames']
        
        if self.split != 'test':
            self.labels = json_data['labels']
            
        print(f'Number of {self.split} images is {len(self.image_names)}')

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):

        ########################################################
        # TODO:                                                #
        # Define the CIFAR10Dataset class:                     #
        #   1. use Image.open() to load image according to the # 
        #      self.image_names                                #
        #   2. apply transform on image                        #
        #   3. if not test set, return image and label with    #
        #      type "long tensor"                              #
        #   4. else return image only                          #
        #                                                      #
        # NOTE:                                                #
        # You will not have labels if it's test set            #
        ########################################################
        image_path = os.path.join(self.dataset_dir, self.image_names[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.split != 'test':
            label = self.labels[index]
            return {'images': image, 'labels': torch.tensor(label, dtype=torch.long)}
            
        return {'images': image}
        
        ###################### TODO End ########################

