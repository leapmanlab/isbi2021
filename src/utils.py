import os
import random

import bio3d_vision as b3d
from bio3d_vision.windowing import gen_conjugate_corners, \
    gen_corner_points

from functools import lru_cache
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from scipy.ndimage.morphology import distance_transform_edt
from skimage.transform import rescale, resize
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import jaccard_score


import torch.nn as nn

from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple


def load_image_dir(image_dir: str,
             n_images: int = -1) -> np.ndarray:
    """Load a sequence of micrograph images from a directory as a single 3D
    array.

    Args:
        image_dir (str): Directory to load images from. Assumes image names are
            of the form f'{z:04}.png' for z in range(n_images).
        n_images (int): Number of images to load. Default is -1 which loads all
            of them.


    Returns:
        (np.ndarray): Array of loaded images stacked along axis 0.

    """
    if n_images < 0:
        n_images = len([f for f in os.listdir(image_dir)
                        if os.path.splitext(f)[1] == '.png'])
    images = [white_balance(plt.imread(os.path.join(image_dir,
                                                    f'{z:04}.png')))
              for z in range(n_images)]
    stack = np.stack(images, axis=0)
    if stack.shape[-1] in [3, 4]:
        stack = stack[..., 0]
    return stack


def load_label_dir(image_dir: str,
             n_images: int = -1) -> np.ndarray:
    """Load a sequence of micrograph labels from a directory as a single 3D
    array.

    Args:
        image_dir (str): Directory to load images from. Assumes image names are
            of the form f'{z:04}.png' for z in range(n_images).
        n_images (int): Number of images to load. Default is -1 which loads all
            of them.


    Returns:
        (np.ndarray): Array of loaded images stacked along axis 0.

    """
    if n_images < 0:
        n_images = len([f for f in os.listdir(image_dir)
                        if os.path.splitext(f)[1] == '.png'])
    images = [(plt.imread(os.path.join(
        image_dir, f'{z:04}.png'))) for z in range(n_images)]
    stack = np.stack(images, axis=0)
    if stack.shape[-1] in [3, 4]:
        stack = (stack[..., -1] > 0).astype(np.int32)
    else:
        stack = (stack > 0).astype(np.int32)
    return stack


def scalable_batch_generator(
        image: np.ndarray,
        label: np.ndarray,
        data_scale: Tuple[float, float, float],
        scaled_image_window_shape: Tuple[int, int, int],
        scaled_label_window_shape: Tuple[int, int, int],
        scaled_window_spacing: Tuple[int, int, int],
        random_windowing: bool = False,
        weight: Optional[np.ndarray] = None,
        do_deformation: bool = False,
        deformation_settings: Optional[Dict[str, Any]] = None,
        return_corners: bool = False,
        return_generators: bool = False):
    """Generate (image, label) pairs or (image, label, weight) triplets of
    data patches from a source image and source ground-truth labels. Using the
    `data_scale` argument, the source data can be scaled up or down before
    batching.

    Args:
        image:
        label:
        data_scale:
        scaled_image_window_shape:
        scaled_label_window_shape:
        scaled_window_spacing:
        random_windowing:
        weight:
        do_deformation:
        deformation_settings:
        return_corners:
        return_generators:

    Returns:

    """
    if do_deformation:
        # Perform elastic deformation
        deformed_image = b3d.deform(
            image,
            deformation_settings,
            random_seed=deformation_settings['seed'])
        # plt.imsave(
        #     f'output/{time.strftime("%y%m%d_%H%M%S.png")}',
        #     deformed_image[0],
        #     cmap='gray')
        deformed_label = b3d.deform(
            label,
            deformation_settings,
            random_seed=deformation_settings['seed'])
        if weight is not None:
            deformed_weight = b3d.deform(
                weight,
                deformation_settings,
                random_seed=deformation_settings['seed'])
        else:
            deformed_weight = None

    else:
        deformed_image = image
        deformed_label = label
        if weight is not None:
            deformed_weight = weight
        else:
            deformed_weight = None

    image_window_shape = [
        rescale(np.zeros(sz), scale=1/sc).shape[0]
        for sz, sc in zip(scaled_image_window_shape, data_scale)]
    label_window_shape = [
        rescale(np.zeros(sz), scale=1 / sc).shape[0]
        for sz, sc in zip(scaled_label_window_shape, data_scale)]
    window_spacing = [
        rescale(np.zeros(sz), scale=1 / sc).shape[0]
        for sz, sc in zip(scaled_window_spacing, data_scale)]

    # Generate a list of corner points for the labels
    label_corner_points = gen_corner_points(
        spatial_shape=deformed_label.shape,
        window_spacing=window_spacing,
        window_shape=label_window_shape,
        random_windowing=random_windowing)

    # Generate a list of corner points for the images
    image_corner_points = gen_conjugate_corners(
        corner_points=label_corner_points,
        window_shape=label_window_shape,
        conjugate_window_shape=image_window_shape)

    # Get window generators
    label_window_generator = scalable_window_generator(
        deformed_label,
        window_shape=label_window_shape,
        scaled_window_shape=scaled_label_window_shape,
        corner_points=label_corner_points,
        interp_order=0)

    image_window_generator = scalable_window_generator(
        deformed_image,
        window_shape=image_window_shape,
        scaled_window_shape=scaled_image_window_shape,
        corner_points=image_corner_points,
        interp_order=0)

    if weight is not None:
        weight_window_generator = scalable_window_generator(
            deformed_weight,
            window_shape=label_window_shape,
            scaled_window_shape=scaled_label_window_shape,
            corner_points=label_corner_points,
            interp_order=0)
        if return_generators:
            return image_window_generator, label_window_generator, \
                weight_window_generator
        else:
            return list(image_window_generator), \
                list(label_window_generator), \
                list(weight_window_generator)
    else:
        if return_corners:
            corners = []
            for z in label_corner_points[0]:
                for x in label_corner_points[1]:
                    for y in label_corner_points[2]:
                        corners.append((z, x, y))
            if return_generators:
                return image_window_generator, label_window_generator, corners
            else:
                return list(image_window_generator), \
                    list(label_window_generator), corners

        else:
            if return_generators:
                return image_window_generator, label_window_generator
            else:
                return list(image_window_generator), \
                   list(label_window_generator)


def scalable_window_generator(
        data_volume: np.ndarray,
        window_shape: Sequence[int],
        scaled_window_shape: Sequence[int],
        corner_points: List[List[int]],
        interp_order: int) -> \
        Generator[np.ndarray, int, None]:
    """

    Args:
        data_volume (np.ndarray): The volume to be preprocessed.
        window_shape (Sequence[int]): The shape of the windows before scaling.
            (x, y) for 2D and (z, x, y) for 3D.
        scaled_window_shape (Sequence[int]): The shape of the windows after
            scaling.
        corner_points (List[List[int], List[int], List[int]]): A list
            specifying the upper-leftmost corner of the windows.
        interp_order (int): Interpolation order for scaling. 0 for
            nearest-neighbor, 1 for linear, 3 for cubic.

    Returns: Generator of windows.
    """
    # Note whether the window is 2D or 3D for later
    window_is_3d = len(window_shape) == 3
    # Make 2D stuff 3D
    if len(window_shape) == 2:
        window_shape = [1] + list(window_shape)

    # Shape of the volumes, and number of dimensions
    # Currently assumes that data will be 2D single channel (2d shape),
    # 3D single channel (3d shape)

    # Is the spatial shape 3D?
    vol_is_3d = data_volume.ndim > 2

    if data_volume.ndim == 2:
        # Add a singleton z spatial dimension
        data_volume = np.expand_dims(data_volume, 0)

    # Volumes' spatial shape
    spatial_shape = data_volume.shape[:]

    # Create windows

    # Calculate the number of windows
    n_windows = 1
    for p in corner_points:
        n_windows *= len(p)

    # Shape of each batch source array. Add in the channel axis:
    # Note the format is either NCXY or NCZXY
    array_shape = [n_windows] + list(window_shape)
    # Remove the z axis if the window is not 3D
    if not window_is_3d or not vol_is_3d:
        array_shape.pop(1)

    # For convenience
    dz = window_shape[0]
    dx = window_shape[1]
    dy = window_shape[2]
    zs = spatial_shape[0]
    xs = spatial_shape[1]
    ys = spatial_shape[2]

    def get_range(n0: int, n1: int, ns: int) -> List[int]:
        """Get a window range along axis n, accounting for reflecting
        boundary conditions when the range is out-of-bounds within the
        source volume.

        Args:
            n0 (int): Window starting point.
            n1 (int): Window ending point.
            ns (int): Source volume size along axis n.

        Returns:
            (List[int]): Window range.

        """

        # Return a range as a list
        def lrange(a, b, n=1) -> List[int]:
            return list(range(a, b, n))

        # Get the in-bounds part of the range
        n_range = lrange(max(0, n0), min(ns, n1))
        # Handle out-of-bounds indices by reflection across boundaries
        if n0 < 0:
            # Underflow
            n_range = lrange(-n0, 0, -1) + n_range
        if n1 > ns:
            # Overflow
            n_range = n_range + lrange(ns - 1, 2 * ns - n1 - 1, -1)

        return n_range

    # Add windows to the batch source arrays
    window_idx = 0
    # Window augmentation parameters
    for x in corner_points[1]:
        for y in corner_points[2]:
            for z in corner_points[0]:

                # Use window_shape-sized windows
                z0 = z
                z1 = z0 + dz
                x0 = x
                x1 = x0 + dx
                y0 = y
                y1 = y0 + dy

                # Compute window ranges
                z_range = get_range(z0, z1, zs)
                x_range = get_range(x0, x1, xs)
                y_range = get_range(y0, y1, ys)

                # Get window extent from the calculated ranges
                window = data_volume.take(z_range, axis=0) \
                    .take(x_range, axis=1) \
                    .take(y_range, axis=2)
                if not window_is_3d or not vol_is_3d:
                    # Remove singleton z dimension for 2D windows
                    window = np.squeeze(window, axis=0)

                window_idx += 1
                scaled_window = resize(
                    window,
                    scaled_window_shape,
                    order=interp_order,
                    preserve_range=True,
                    anti_aliasing=False).astype(window.dtype)
                yield scaled_window

    return None


def white_balance(arr: np.ndarray, perc: float = 0.05) -> np.ndarray:
    """Perform white balancing on a grayscale image array.

    Args:
        arr (np.ndarray): Grayscale image array.
        perc (float): Image value cutoff percentile.

    Returns:
        (np.ndarray): White balanced image array.

    """
    mi = np.percentile(arr, perc)
    ma = np.percentile(arr, 100 - perc)
    return np.float32(np.clip((arr - mi) / (ma - mi), 0, 1))


class PlateletDataset(Dataset):

    def __init__(self, images, labels, weight=None, train=True):

        self.images = images
        self.labels = labels
        self.train = train
        self.weight = weight

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        weight = None
        if self.weight is not None:
            weight = self.weight[idx]
        if self.train:
            image, label, weight = self.transform(image, label, weight)
            return image, label, weight
        else:
            return image, label

    def transform(self, image, mask, weight=None):
        for i in [1, 2]:
            # Random flipping
            if random.random() > 0.5:
                image = np.flip(image, i).copy()
                mask = np.flip(mask, i).copy()
                if weight is not None:
                    weight = np.flip(weight, i).copy()
        # Transform to tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        if weight is not None:
            weight = torch.from_numpy(weight)
            return image, mask, weight
        else:
            return image, mask


def imshow(images: Sequence[np.ndarray],
           figsize: Sequence[int],
           plot_settings: Optional[Sequence[Dict[str, Any]]] = None,
           layout: Optional[Tuple[int, int]] = None,
           frame: bool = True) -> None:
    """Simplify showing one or more images
    Args:
        images:
        figsize:
        plot_settings:
        layout:
        frame:
    Returns:
    """
    if not isinstance(images, Sequence):
        images = [images]

    if layout is None:
        layout = (1, len(images))

    f, axs = plt.subplots(
        nrows=layout[0],
        ncols=layout[1],
        figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, ax in enumerate(axs):
        if plot_settings is not None:
            ax.imshow(images[i], **plot_settings[i], extent=(0, 1, 1, 0))
        else:
            ax.imshow(images[i], extent=(0, 1, 1, 0))
        ax.axis('tight')
        if frame:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        else:
            ax.axis('off')

    return f


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        torch.nn.init.xavier_uniform_(m.weight)


def center_crop(X, target_shape):
    X = np.squeeze(X)
    target_height, target_width = target_shape
    _, height, width = X.shape

    diff_x = (height - target_height) // 2
    diff_y = (width - target_width) // 2

    return X[:,
           diff_x: (diff_x + target_height),
           diff_y: (diff_y + target_width)
           ]


class DuplicateFilter:
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

def log_weights(experiment, model):
    """i = 1
    for conv_layer in model.first_set:
        if type(conv_layer) == nn.Conv3d:
            values = conv_layer.weight.data
            experiment.log_histogram_3d(values=values,
                                        name=f"conv_{i}_weights")
            i += 1
    for conv_layer in model.second_set:
        if type(conv_layer) == nn.Conv3d:
            values = conv_layer.weight.data
            experiment.log_histogram_3d(values=values,
                                        name=f"conv_{i}_weights")
            i += 1
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            experiment.log_histogram_3d(values=param.data,
                                        name=f"{name}")


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

@lru_cache(maxsize=128)
def memoized_distance_transform(
        shape: Sequence[int],
        distance_scale: Optional[Sequence[int]] = None) -> np.ndarray:
    """

    Args:
        shape:
        distance_scale

    Returns:
        (np.ndarray)
    """
    if not distance_scale:
        distance_scale = [1 for s in shape]

    expanded_shape = [s + 2 for s in shape]

    arr_padded = np.zeros(expanded_shape)
    center_slice = (slice(1, -1),) * len(shape)
    arr_padded[center_slice] = 1
    dist_padded = distance_transform_edt(arr_padded, sampling=distance_scale)
    return dist_padded[center_slice]


def stitch(model,
           eval_images,
           eval_labels,
           big_vol_shape,
           eval_label_corners,
           windowing_params,
           net_is_3d=False,
           n_classes=7,
           device="cuda",
           channels=1):

    if net_is_3d:
        distance_scale = (4, 1, 1)
    else:
        distance_scale = (1, 1)

    output_shape = windowing_params['label_window_shape']
    prob_map_update_dist = np.zeros(big_vol_shape)
    prob_shape = [n_classes] + list(big_vol_shape)
    prob_maps = np.zeros(prob_shape)
    for i, (X, y, c) in enumerate(zip(eval_images,
                                      eval_labels,
                                      eval_label_corners
                                      )
                                  ):
        X = torch.from_numpy(np.expand_dims(X, axis=0))
        if channels==3:
            X = X.repeat(1, 3, 1, 1)  # For deeplab
        prediction = model(X.to(device)).cpu().detach()
        classes = torch.argmax(prediction, dim=1)
        patch_prob = torch.squeeze(prediction).numpy()
        patch_dist = memoized_distance_transform(patch_prob.shape[1:],
                                                 distance_scale)
        patch_corner = c

        # if net_is_3d:
        z0 = patch_corner[0]
        z1 = z0 + output_shape[0]
        x0 = patch_corner[1]
        x1 = x0 + output_shape[1]
        y0 = patch_corner[2]
        y1 = y0 + output_shape[2]
        region = (slice(z0, z1), slice(x0, x1), slice(y0, y1))
        """
        else:
            z0 = patch_corner[0]
            x0 = patch_corner[1]
            x1 = x0 + output_shape[1]
            y0 = patch_corner[2]
            y1 = y0 + output_shape[2]
            if ndim_data > 2:
                region = [slice(z0), slice(x0, x1), slice(y0, y1)]
            else:
                region = [slice(x0, x1), slice(y0, y1)]
        """

        parts_to_update = prob_map_update_dist[region] < patch_dist

        padded_parts_to_update = np.zeros_like(prob_map_update_dist,
                                               dtype=np.bool)
        padded_parts_to_update[region] = parts_to_update

        prob_maps[:, padded_parts_to_update] = patch_prob[:,
                                               np.squeeze(parts_to_update)]
        prob_map_update_dist[padded_parts_to_update] = \
            patch_dist[np.squeeze(parts_to_update)]

        """
        if i % 100 == 0:
            images = (np.squeeze(X.cpu().numpy()),
                      np.squeeze(classes.cpu().numpy()),
                      np.squeeze(y))
            fig = imshow(images, (15, 5), plot_settings)
            plt.show()
        """
    return prob_maps


def evaluate(
        true: np.ndarray,
        prediction: np.ndarray,
        do_nonzero_masking: bool = False,
        labels = range(7),
        organelle_labels = range(2,7)) -> Dict[str, Any]:
    """

    Args:
        true:
        prediction:
        do_nonzero_masking:

    Returns:
        (Dict[str, Any]):

    """
    if do_nonzero_masking:
        eval_mask = true > 0
        true = true[eval_mask]
        prediction = prediction[eval_mask]
    else:
        true = true.flatten()
        prediction = prediction.flatten()

    miou = float(sklearn.metrics.jaccard_score(
        y_true=true,
        y_pred=prediction,
        average='macro',
        labels=labels))
    ari = float(sklearn.metrics.adjusted_rand_score(
        labels_true=true,
        labels_pred=prediction))
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=true,
        y_pred=prediction)
    normalized_confusion_matrix = \
        confusion_matrix.astype(np.float) / \
        confusion_matrix.sum(axis=1)[:, np.newaxis]
    organelle_miou = float(sklearn.metrics.jaccard_score(
        y_true=true,
        y_pred=prediction,
        average='macro',
        labels=organelle_labels))

    results = {
        'classes_present': unique_labels(true, prediction).tolist(),
        'miou': miou,
        'ari': ari,
        'confusion_matrix': normalized_confusion_matrix.tolist(),
        'organelle_miou': organelle_miou}
    return results
