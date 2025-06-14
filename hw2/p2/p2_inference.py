# ============================================================================
# File: p2_inference.py
# Date: 2025-03-11
# Author: TA
# Description: Load pre-trained model and perform inference on test set.
# ============================================================================
import os
import sys
import time
import argparse
import numpy as np
import torch

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import write_csv

def main():

    # Initialize ArgumentParser to parse command-line arguments
    parser = argparse.ArgumentParser()

    # Add argument for the test dataset directory
    # `--test_datadir`: path to the test data directory (default: '../hw2_data/p2_data/val/')
    parser.add_argument('--test_datadir',
                        help='test dataset directory',
                        type=str,
                        default='../hw2_data/p2_data/val/')

    # Add argument for the model type ('mynet' or 'resnet18')
    # `--model_type`: which model to use for inference (default: 'resnet18')
    parser.add_argument('--model_type',
                        help='mynet or resnet18',
                        type=str,
                        default='resnet18')

    # Add argument for output CSV file path
    # `--output_path`: where to save the output predictions (default: './output/pred.csv')
    parser.add_argument('--output_path',
                        help='output csv file path',
                        type=str,
                        default='./output/pred.csv')

    # Parse the arguments
    args = parser.parse_args()

    # Extract the parsed arguments
    model_type = args.model_type  # Type of model to use
    test_datadir = args.test_datadir  # Directory containing the test dataset
    output_path = args.output_path  # Path where predictions will be saved

    # Check if a GPU is available; otherwise, fall back to CPU
    # `torch.cuda.is_available()` checks if CUDA (GPU support) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)  # Print the device being used (CPU or GPU)

    ##### MODEL #####
    # Load the pre-trained model based on the model_type argument

    # For 'mynet' model
    if model_type == 'mynet':
        model = MyNet()  # Initialize MyNet model
        # Load the model weights from the checkpoint for 'mynet'
        model.load_state_dict(torch.load('./checkpoint/mynet_best.pth', 
                                         map_location=torch.device('cpu')))
    # For 'resnet18' model
    elif model_type == 'resnet18':
        model = ResNet18()  # Initialize ResNet18 model
        # Load the model weights from the checkpoint for 'resnet18'
        model.load_state_dict(torch.load('./checkpoint/resnet18_best.pth', 
                                         map_location=torch.device('cpu')))
    else:
        # Raise an error if an unknown model type is provided
        raise NameError('Unknown model type')

    # Move the model to the device (GPU or CPU)
    model.to(device)

    ##### DATALOADER #####
    # Get the test dataloader (returns an iterable over the test dataset)
    test_loader = get_dataloader(test_datadir, batch_size=1, split='test')

    ##### INFERENCE #####
    # List to store the predictions for each image
    predictions = []
    # Set the model to evaluation mode (disables dropout, batch normalization behavior)
    model.eval()

    # Disable gradient calculation (we're doing inference, not training)
    with torch.no_grad():
        # Record the start time for testing
        test_start_time = time.time()

        #############################################################
        # Inference loop:                                         #
        # For each batch in the test loader, forward the image through the model
        # and append the predicted label to the 'predictions' list. #
        #############################################################
        
        for batch in test_loader:
            # Extract the image data from the batch , check get_item() function in dataset.py
            images = batch['images']
            # Move the image data to the correct device (GPU or CPU)
            images = images.to(device)
            # Forward pass: get the model's outputs (logits), automatically call model.forward()
            outputs = model(images)
            # Get the predicted class (index with the highest score)
            _, predicted = torch.max(outputs, 1)
            # Append the predicted class label (as an integer) to the predictions list
            predictions.append(predicted.item())
            
        ######################### TODO End ##########################

    # Calculate the time taken for testing
    test_time = time.time() - test_start_time
    print()  # Print a new line
    # Print the total time taken for inference
    print(f'Finish testing {test_time:.2f} sec(s), dumps result to {output_path}')

    ##### WRITE RESULT #####
    # Write the predictions to a CSV file
    write_csv(output_path, predictions, test_loader)

if __name__ == '__main__':
    main()

    