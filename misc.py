"""
Filename:        misc.py
Author:          Jia-Yi LI
Last Modified:   2024-12-20
Version:         1.0
Description:     Miscellaneous utilities.
                 Copyright (c) 2024 Nanyang Technological University
"""

import numpy as np
from skimage.measure import block_reduce
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed


# Generates a 2D array with randomized rows within a range
def getSweepingArrayRandom(startV=-3, stopV=3, stepV=0.02):
    rows = np.arange(startV, stopV + stepV, stepV)
    numofrows = int(abs(startV - stopV) / stepV)
    array_2d = np.transpose(np.tile(rows, (numofrows, 1)))
    array_2d = np.delete(array_2d, np.s_[16:], axis=1)
    np.random.seed(42)
    np.random.shuffle(array_2d)
    return array_2d

# Applies pooling operation to an image array
def pool(img, block_size=(2, 2), method='max_pool', verbose=True):
    img_pool = np.empty((img.shape[0], int(img.shape[1] / block_size[0]), int(img.shape[2] / block_size[1])))

    if method == 'max_pool':
        func = np.max
    elif method == 'avg_pool':
        func = np.average

    for i in range(img.shape[0]):
        img_pool[i] = block_reduce(img[i], block_size, func)
    
    if verbose:
        print(f"{method}ing from {img.shape} to {img_pool.shape}")

    return img_pool

# Computes ANOVA between two groups
def compute_anova_pair(group1, group2):
    f_value, p_value = stats.f_oneway(group1, group2, axis=0)
    return f_value, p_value

# Performs pairwise ANOVA on input data
def anova(arr, verbose=True):
    num_groups = arr.shape[1]
    p_values = np.zeros((num_groups, num_groups))
    f_values = np.zeros_like(p_values)

    pairs = [(i, i + j) for i in range(num_groups) for j in range(1, num_groups - i)]

    if verbose:
        results = Parallel(n_jobs=-1)(delayed(compute_anova_pair)(arr[:, i], arr[:, j]) for i, j in tqdm(pairs))
    else:
        results = Parallel(n_jobs=-1)(delayed(compute_anova_pair)(arr[:, i], arr[:, j]) for i, j in pairs)

    ns_count = 0
    for idx, (i, j) in enumerate(pairs):
        f_value, p_value = results[idx]
        p_values[j, i] = p_value
        f_values[j, i] = f_value
        if p_value > 0.1:
            ns_count += 1

    g_count = len(pairs)
    ns_percentage = ns_count / g_count * 100

    if verbose:
        print(f"Not significant pairs number: {ns_count} vs {g_count}\nPercentage: {ns_percentage:.2f}%")

    return f_values, p_values, ns_percentage

if __name__ == "__main__":
    result = getSweepingArrayRandom(-3, 0, 0.02)
    print(result)
