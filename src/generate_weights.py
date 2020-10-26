import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as skm

from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects

from PIL import Image

lcimb_dir = os.path.join('..', '..', '..', '..')


def mistake_correction(
        idx: int,
        blur_sigma: float = 5,
        pred_error_mode: str = 'both',
        min_positive_size: int = 0,
        train_dir: str = os.path.join(lcimb_dir, 'data', 'platelet-membrane', 'sample', 'train'),
        raw_label_dir: str = 'label-raw',
        correct_label_dir: str = 'label') -> np.ndarray:
    """Generate error weights by comparing raw predictions with corrections and
    upweighting areas where predictions are wrong.

    Args:
        idx (int): Z-slice index to load.
        blur_sigma (float): Apply a Gaussian blur to the error mask to upweight
            neighboring regions, with this standard deviation.
        pred_error_mode (str): Prediction error mode. Currently 'positive',
            'negative', or 'both' are supported. The 'positive' mode penalizes
            only false positives, 'negative' penalizes only false negatives,
            and 'both' penalizes both.
        min_positive_size (int): Minimum size of positive regions to include.
            Anything smaller is set to negative.
        train_dir (str): Directory containing the training data.
        raw_label_dir (str): Directory inside `train_dir` containing the raw
            label files.
        correct_label_dir (str): Directory inside `train_dir` containing the
            corrected label files.

    Returns (np.ndarray): Weight array, scaled so that the max is 1.

    """
    raw_label_dir = os.path.join(train_dir, raw_label_dir)
    raw_label_file = os.path.join(raw_label_dir, f'{idx:04}.png')
    raw_label = (plt.imread(raw_label_file)[..., 3] > 0).astype(np.float32)

    correct_label_dir = os.path.join(train_dir, correct_label_dir)
    correct_label_file = os.path.join(correct_label_dir, f'{idx:04}.png')
    correct_label = (plt.imread(correct_label_file)[..., 3] > 0).astype(np.float32)

    if pred_error_mode == 'both':
        label_diff = correct_label != raw_label
    elif pred_error_mode == 'negative':
        label_diff = (correct_label - raw_label) > 0
    elif pred_error_mode == 'positive':
        label_diff = (correct_label - raw_label) < 0
    else:
        raise ValueError(
            f'pred_error_mode value "{pred_error_mode}" not recognized')

    if min_positive_size > 0:
        remove_small_objects(label_diff, min_positive_size, in_place=True)

    if blur_sigma > 0:
        blurred_diff = gaussian_filter(label_diff.astype(float), sigma=blur_sigma)
        blurred_diff = blurred_diff / blurred_diff.max()
        return blurred_diff
    return label_diff.astype(float) / label_diff.max()


def cell_distance(idx: int):
    # Load cell membrane boundary data and cell instance data
    boundary_dir = '/home/matt/Dropbox/nibib/data/platelet-membrane/sample/train/label'
    instance_dir = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments/instance_labels/local/train data/cell instance xy/final masks'

    boundary_file = os.path.join(
        boundary_dir,
        f'{idx:04}.png')
    boundary = plt.imread(boundary_file)[..., 3] > 0

    instance_file = os.path.join(
        instance_dir,
        f'fixed_cell_instance_labels{idx:08}.png')
    instance_im = Image.open(instance_file)
    instance = (np.array(instance_im) > 0).astype(np.uint8)

    # Get connected components of the boundary file that don't correspond to
    # cell membrane regions
    all_components = skm.label(boundary, background=-1)
    nonmembrane_ids = [i for i in np.unique(all_components)
                       if boundary[all_components == i].mean() < 0.5]

    # Fill in nonmembrane connected components as "cell" or "background"
    classes = np.copy(boundary).astype(np.uint8)
    for i in nonmembrane_ids:
        component_mask = all_components == i
        if component_mask.sum() < 100:
            classes[component_mask] = 0
        else:
            classes[component_mask] = 2 * (instance[component_mask].mean() > 0.5)

    plt.imsave('classes.png', classes, cmap='magma')


if __name__ == '__main__':
    blur = mistake_correction(1, 5)
    # print(blur.shape)
    plt.imshow(blur)
    plt.show()