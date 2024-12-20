"""
Filename:        usrUtilis.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Utilities for dataset process.
License:         MIT License
                 Copyright (c) 2024 Nanyang Technological University
"""
import os

def pad_and_rename_files(directory):
    # List the files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file
    for filename in files:
        # Ignore .DS_Store file
        if filename == ".DS_Store":
            continue
        # Split the filename by '-' and '.' to extract the number part
        parts = filename.split('-')
        number_part = parts[-1].split('.')[0]
        extension = '.' + parts[-1].split('.')[-1]
        
        # Pad the numeric part with leading zeros
        padded_number = number_part.zfill(4)
        
        # Construct the new filename
        new_filename = parts[0] + '-' + padded_number + extension
        
        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        
        # print(f'Renamed {filename} to {new_filename}')


def label_to_text(label, map = 'mnist'):
    # Dictionary mapping label numbers to text labels
    if map == 'fmnist':
        label_map = {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        }
    elif map == 'mnist':
                label_map = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9'
        }

    # Check if the label is in the valid range
    if label in label_map:
        return label_map[label]
    else:
        raise ValueError("Label number must be between 0 and 9.")