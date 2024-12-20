"""
Filename:        HAR_toPulse.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Translate HAR videos to pulse input.
License:         MIT License
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
from Data.dataset.HAR import readWeizmannRevisited as HAR
import misc,os

saveScript = False
saveLabel = False

root_drc = os.path.abspath('DIRC_TO_PULSE_DATA_OUTPUT')

frame_N = 8
block_size = (12,12)

video, labels = HAR.ctDatasetTenClas(frames_per_clip = frame_N)
video = np.reshape(video, (video.shape[0], video.shape[1], 144, 180))

videos = np.empty((video.shape[0], video.shape[1], int(144/block_size[0]), int(180/block_size[1])))

for i in range(frame_N):
    videos[:,i, :, :] = misc.pool(video[:,i, :, :], block_size, method = 'avg_pool') # max_pool, avg_pool

print(f"Video shape after pooling is {videos.shape}")

X_train = videos
y_train = labels

RC_input = np.zeros((20, frame_N, videos.shape[2], videos.shape[3]))
har_img = np.zeros((20, frame_N, video.shape[2], video.shape[3]))

for i in range (10):
    chosen_label = i
    indices = np.where(y_train == chosen_label)[0]
    prng = np.random.RandomState(42)
    random_index = prng.choice(indices)
    RC_input[i] = X_train[random_index]
    har_img[i] = video[random_index]
for i in range (10,20):
    chosen_label = i-10
    indices = np.where(y_train == chosen_label)[0]
    prng = np.random.RandomState(0)
    random_index = prng.choice(indices)
    RC_input[i] = X_train[random_index]
    har_img[i] = video[random_index]
RC_input = np.transpose(RC_input, (0, 2, 3, 1))
RC_input = np.reshape(RC_input, (RC_input.shape[0], RC_input.shape[1]* RC_input.shape[2], RC_input.shape[3]))
RC_input = np.reshape(RC_input, (RC_input.shape[0]* RC_input.shape[1], RC_input.shape[2]))
print(RC_input.shape)

i = 0
id_HAR = 2405171541
rc_input = (RC_input-0.5)*6

puls_in = np.transpose(rc_input)
puls_in = np.reshape(puls_in, (8, 20, 12, 15))