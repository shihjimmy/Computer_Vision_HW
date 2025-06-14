# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()

        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        # Define the CNN architecture using nn.Sequential
        # Using nn.Sequential to define the layers of the convolutional network in sequence
        self.cnn = nn.Sequential(
            # First Convolutional layer: Input channels = 3 (RGB), Output channels = 32 (32 kernels to convolution)
            # Apply 3x3 convolution kernel with padding 1 to preserve spatial dimensions
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  

            # Batch Normalization for 32 channels
            # Normalize activations to have zero mean and unit variance, which helps stabilize training
            nn.BatchNorm2d(32),  

            # ReLU Activation function
            # Introduce non-linearity by setting all negative values to zero and keeping positive values as is
            nn.ReLU(inplace=True),  

            # Second Convolutional layer: Input channels = 32, Output channels = 32
            # Use 3x3 convolution kernel with padding 1
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  

            # Batch Normalization for 32 channels
            # Again normalize the output of the convolutional layer
            nn.BatchNorm2d(32),  

            # ReLU Activation function
            nn.ReLU(inplace=True),  

            # Max Pooling layer: Reduces spatial dimensions by taking the maximum value in a 2x2 window
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x32 --> 32x16x16

            # Third Convolutional layer: Input channels = 32, Output channels = 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  

            # Batch Normalization for 64 channels
            nn.BatchNorm2d(64),  

            # ReLU Activation function
            nn.ReLU(inplace=True),  

            # Fourth Convolutional layer: Input channels = 64, Output channels = 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  

            # Batch Normalization for 64 channels
            nn.BatchNorm2d(64),  

            # ReLU Activation function
            nn.ReLU(inplace=True),  

            # Max Pooling layer: Reduces spatial dimensions again
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x16x16 --> 64x8x8

            # Fifth Convolutional layer: Input channels = 64, Output channels = 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  

            # Batch Normalization for 128 channels
            nn.BatchNorm2d(128),  

            # ReLU Activation function
            nn.ReLU(inplace=True),  

            # Sixth Convolutional layer: Input channels = 128, Output channels = 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  

            # Batch Normalization for 128 channels
            nn.BatchNorm2d(128),  

            # ReLU Activation function
            nn.ReLU(inplace=True),  

            # Max Pooling layer: Reduces spatial dimensions one last time
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x8x8 --> 128x4x4
        )

        # Define the fully connected (FC) layers for classification
        # nn.Sequential allows us to define the fully connected layers in sequence
        self.fc = nn.Sequential(
            # First Fully Connected layer: Input features = 128*4*4, Output features = 512
            # This layer takes the flattened output of the convolutional layers and maps it to 512 features
            nn.Linear(128*4*4, 512),  

            # Batch Normalization for 512 features
            # Normalize the fully connected layer output to stabilize learning
            nn.BatchNorm1d(512),  

            # ReLU Activation function
            # Introduce non-linearity again to allow the network to learn more complex patterns
            nn.ReLU(inplace=True),  

            # Second Fully Connected layer: Input features = 512, Output features = 10 (for classification)
            # The final output will correspond to the 10 class predictions (e.g., CIFAR-10 classes)
            nn.Linear(512, 10),  
        )

    def forward(self, x):
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        # Forward pass through the convolutional layers
        # Apply the CNN layers defined in self.cnn to the input 'x'
        out = self.cnn(x)  

        # Flatten the output from the convolutional layers
        # Reshape the 4D output from the convolutional layers (batch_size, channels, height, width)
        # into a 2D tensor (batch_size, flattened_features) to feed into the fully connected layers
        out = out.view(out.size(0), -1)  

        # Forward pass through the fully connected layers
        # Apply the fully connected layers defined in self.fc
        out = self.fc(out)  

        # Return the final output (class predictions)
        # The output is a tensor of size (batch_size, 10), representing class scores for each input image
        return out

    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################
 
        # Try to load the pretrained weights (for models in PyTorch 1.10.1 and earlier, use pretrained=False)
        # self.resnet = models.resnet18(weights=None)  # Python 3.8 with torch 2.2.1
        # self.resnet = models.resnet18(pretrained=False)  # Python 3.6 with torch 1.10.1

        # If pretrained=True, it loads the weights that were pretrained on ImageNet.
        # ResNet18 model with pretrained weights
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 32x32 images (rather than the typical 224x224 images for ImageNet).
        # The original ResNet18 uses a 7x7 kernel with a stride of 2. Here, we're using a 3x3 kernel with stride=1.
        # This is to better handle smaller input images like CIFAR-10, which are 32x32.
        # We're setting bias=False because BatchNorm will handle the bias term.
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the first max pooling layer and replace it with Identity().
        # The maxpool layer in ResNet18 operates on 224x224 images, but CIFAR-10 images are 32x32.
        # So we remove it to avoid reducing spatial dimensions too much.
        # Identity() is a no-operation layer, which will effectively skip this layer.
        self.resnet.maxpool = nn.Identity()

        # Modify the final fully connected layer to output 10 classes instead of 1000.
        # ResNet18 originally outputs 1000 classes (for ImageNet), but CIFAR-10 has only 10 classes.
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)
    
