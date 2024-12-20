"""
Filename:        readWeizmannRevisited.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Read HAR dataset.
License:         MIT License
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
import scipy.io

# Load the dataset
# Replace with the actual path to your data file
data_path = "Data/dataset/HAR/"
mat = scipy.io.loadmat(data_path + 'classification_masks.mat')['original_masks']
data = mat[0, 0]

# Dictionary to map label suffixes to numeric labels
label_mapping = {
    'bend': 0,
    'jack': 1,
    'jump': 2,
    'run': 3,
    'run1': 3,
    'run2': 3,
    'side': 4,
    'skip': 5, 'skip1': 5, 'skip2': 5,
    'wave1': 6,
    'wave2': 7,
    'walk': 8, 'walk1': 8, 'walk2': 8,
    'pjump': 9
}

label_mapping_for_print = {
    'Bend': 0,
    'Jumping jack': 1,
    'Jump': 2,
    'Run': 3,
    'Gallop sideways': 4,
    'Skip': 5,
    'Wave one hand': 6,
    'Wave two hands': 7,
    'Walk': 8,
    'Jump in place on two legs': 9
}

def videoLabel(idx):
    """
    Retrieve the label name corresponding to a numeric label index.

    Args:
        idx (int): Numeric label index.

    Returns:
        str: Corresponding label name.
    """
    return list(label_mapping_for_print.keys())[list(label_mapping_for_print.values()).index(idx)]

def ctDatasetTenClas(frames_per_clip=4, step_size=1):
    """
    Create clips and their corresponding labels from the dataset.

    Args:
        frames_per_clip (int): Number of frames in each clip.
        step_size (int): Step size for creating clips.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of clips.
            - numpy.ndarray: Array of corresponding labels.
    """
    clips_list = []
    labels_list = []

    for field_name in data.dtype.names:
        video_data = data[field_name]
        num_frames = video_data.shape[-1]

        for frame_start in range(0, num_frames - frames_per_clip + 1, step_size):
            clip_frames = video_data[..., frame_start:frame_start + frames_per_clip]
            clips_list.append(clip_frames)
            suffix = field_name.split('_')[-1]
            labels_list.append(label_mapping[suffix])

    clips_array = np.array(clips_list)
    labels_array = np.array(labels_list)
    clips_array = np.transpose(clips_array, (0, 3, 1, 2))
    clips_array = np.reshape(clips_array, (clips_array.shape[0], clips_array.shape[1], clips_array.shape[2] * clips_array.shape[3]))

    return clips_array, labels_array
