"""Segment a large volume with stitched-together windows.

"""
import importlib.util
import json
import os
import sys
import time

import bio3d_vision as b3d
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.transform import rescale, resize

sys.path.append('..')
# noinspection PyUnresolvedReferences
from src.utils import gen_corner_points, gen_conjugate_corners, \
    scalable_window_generator, white_balance, load_image_dir

from typing import Any, Dict, Generator, Optional, Sequence, Tuple, Union


def segment_online(
        image: np.ndarray,
        model: nn.Module,
        model_config: Dict[str, Any],
        save_file: Optional[str] = None,
        save_cmap: str = 'jet',
        window_spacing: Optional[Tuple[int, int, int]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5) -> np.ndarray:
    """Segment an image (already loaded into memory as an np.ndarray) using
    a model that is also already loaded into memory.

    Args:
        image (np.ndarray): Image to segment.
        model (nn.Module): Segmentation model.
        model_config (Dict[str, Any]): Dict containing model i/o info. Keys:
            shape_in: Model input shape.
            shape_out: Model output shape.
            data_scale: Data zoom along each axis.
        save_file (Optional[str]): If provided, save the output segmentation
            to this file.
        save_cmap (str): Name of a matplotlib colormap. Default is 'jet'.
        window_spacing (Optional[Tuple[int, int, int]): Spacing between
            consecutive windows used to tile up the input image file. Default is
            the model's output window size, meaning consecutive windows do
            not overlap. With overlapping windows, some voxels will receive
            multiple predictions, which are averaged before choosing the final
            most likely segmentation class.
        device (Optional[str]): If provided, specify which PyTorch device to
            use. Default is the first available GPU.
        threshold (float): Detection threshold.

    Returns:
        (np.ndarray): Segmented image

    """
    ''' Setup '''

    if window_spacing is None:
        window_spacing = model_config['shape_out']
    model_config['spacing'] = window_spacing
    n_classes = 1

    ''' Run segmentation '''

    class_probabilities = prob_maps_online(
        image=image,
        model=model,
        model_config=model_config,
        n_classes=1,
        device=device)

    segmentation = class_probabilities > threshold

    ''' Save segmentation, if specified '''

    if save_file is not None:
        if n_classes == 1:
            seg_rescaled = (255 * segmentation).astype(np.uint8)
        else:
            seg_rescaled = (255 / (n_classes - 1) * segmentation).astype(
                np.uint8)

        save_ext = os.path.splitext(save_file)[1].lower()
        if 'tif' in save_ext:
            cmap = tif_cmap(save_cmap)
            tif.imsave(save_file, seg_rescaled, colormap=cmap, compress=7)
        elif 'npy' in save_ext:
            cmap = plt.get_cmap(save_cmap)
            output = (255 * cmap(seg_rescaled))[..., :3].astype(np.uint8)
            np.save(save_file, output)
        elif 'png' in save_ext:
            output_shape = np.squeeze(seg_rescaled).shape
            output = np.zeros((*output_shape, 4), dtype=np.uint8)
            output[..., 2] = seg_rescaled
            output[..., 3] = (255 * (seg_rescaled > 0)).astype(np.uint8)
            print(output.shape)
            plt.imsave(save_file, output)
        else:
            raise ValueError(f'Save extension {save_ext} not recognized.')

    return segmentation


def segment(
        image_file: str,
        pth_files: Union[str, Sequence[str]],
        save_file: Optional[str] = None,
        ensemble_mode: str = 'max',
        save_cmap: str = 'jet',
        window_spacing: Optional[Tuple[int, int, int]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        return_probs: bool = False,
        verbose: bool = False) -> np.ndarray:
    """Segment an image one or more PyTorch models.

        Args:
            image_file (str): Image file to segment.
            pth_files (Union[str, Sequence[str]]): Name of the .pth files to
                load. This function assumes that the directory containing the
                .pth file also contains excatly one model config file (.json)
                and exactly one model source file (.py).
            save_file (Optional[str]): If provided, save the output segmentation
                to this file.
            ensemble_mode (str): Ensembling mode. 'mean' to average the
                predictions, 'max' to take the max of each prediction.
            save_cmap (str): Name of a matplotlib colormap. Default is 'jet'.
            window_spacing (Optional[Tuple[int, int, int]): Spacing between
                consecutive windows used to tile up the input image file. Default is
                the the model's output window size, meaning consecutive windows do
                not overlap. With overlapping windows, some voxels will receive
                multiple predictions, which are averaged before choosing the final
                most likely segmentation class.
            device (Optional[str]): If provided, specify which PyTorch device to
                use. Default is the first available GPU.
            threshold (float): Detection threshold.
            return_probs (bool): If True, skip the segmentation and just return
                the class probability map.
            verbose (bool): If True, print verbose segmentation and ensembling
                information and timing.

        Returns:
            (np.ndarray): Segmentation of the input image.

        """
    time_start = time.time()
    if isinstance(pth_files, str):
        pth_files = [pth_files]

    model_dirs = [os.path.dirname(os.path.realpath(pth_file))
                  for pth_file in pth_files]
    pth_stubs = [os.path.basename(pth_file) for pth_file in pth_files]

    if verbose:
        print(f'Segmenting with {len(pth_files)} nets')

    prob_map_list = []
    for i, (model_dir, pth_stub) in enumerate(zip(model_dirs, pth_stubs)):
        time0 = time.time()
        prob_map_list.append(_segment(
            image_file,
            model_dir,
            None,
            save_cmap,
            window_spacing,
            device,
            threshold,
            pth_stub,
            True))
        time1 = time.time()
        if verbose:
            print(f'Computed prob map {i} ({time1-time0:.1f} s)')
    prob_maps = np.stack(prob_map_list, axis=-1)

    if ensemble_mode == 'mean':
        prob_map = prob_maps.mean(axis=-1)
    elif ensemble_mode == 'max':
        prob_map = prob_maps.max(axis=-1)
    else:
        raise ValueError(f'Ensemble mode {ensemble_mode} not recognized')

    if return_probs:
        if save_file is not None:
            probs_rescaled = (255 * prob_map).astype(np.uint8)
            print(len(np.unique(probs_rescaled)))
            output_shape = np.squeeze(probs_rescaled).shape
            output = np.zeros((*output_shape, 4), dtype=np.uint8)
            output[..., 2] = probs_rescaled
            output[..., 3] = (255 * (probs_rescaled > 0)).astype(np.uint8)

            save_ext = os.path.splitext(save_file)[1].lower()
            if 'tif' in save_ext:
                tif.imsave(save_file, output, compress=7)
            elif 'png' in save_ext:
                plt.imsave(save_file, output)
        if verbose:
            print(f'Returning probability maps '
                  f'({time.time() - time_start:.1f} s)')
        return prob_map

    segmentation = prob_map > threshold
    segmentation_uint8 = (255 * (prob_map > threshold)).astype(np.uint8)

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        seg_shape = np.squeeze(segmentation_uint8).shape
        seg_img = np.zeros((*seg_shape, 4), dtype=np.uint8)
        seg_img[..., 2] = segmentation_uint8
        seg_img[..., 3] = (255 * (segmentation_uint8 > 0)).astype(np.uint8)
        if 'tif' in os.path.splitext(save_file)[1]:
            tif.imsave(save_file, seg_img, compress=7)
        else:
            plt.imsave(save_file, seg_img)

    if verbose:
        print(f'Returning segmentation ({time.time() - time_start:.1f} s)')

    return segmentation


def _segment(
        image_file: str,
        model_dir: str,
        save_file: Optional[str] = None,
        save_cmap: str = 'jet',
        window_spacing: Optional[Tuple[int, int, int]] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        pth_file: str = 'best_weights.pth',
        return_probs: bool = False) -> np.ndarray:
    """Segment an image using a PyTorch model.

    Args:
        image_file (str): Image file to segment.
        model_dir (str): Directory containing a PyTorch model weights, source
            code, and model config information.
        save_file (Optional[str]): If provided, save the output segmentation
            to this file.
        save_cmap (str): Name of a matplotlib colormap. Default is 'jet'.
        window_spacing (Optional[Tuple[int, int, int]): Spacing between
            consecutive windows used to tile up the input image file. Default is
            the the model's output window size, meaning consecutive windows do
            not overlap. With overlapping windows, some voxels will receive
            multiple predictions, which are averaged before choosing the final
            most likely segmentation class.
        device (Optional[str]): If provided, specify which PyTorch device to
            use. Default is the first available GPU.
        threshold (float): Detection threshold.
        pth_file (str): Name of the .pth file to load from the `model_dir`.
        return_probs (bool): If True, skip the segmentation and just return
            the class probability map.

    Returns:
        (np.ndarray): Segmentation of the input image.

    """
    ''' Load necessary info from the model dir '''

    # Model config info
    json_files = [f for f in os.listdir(model_dir)
                  if os.path.splitext(f)[1] == '.json']
    if len(json_files) != 1:
        raise NameError(f'{model_dir} must contain exactly one .json file.')
    json_file = json_files[0]
    with open(os.path.join(model_dir, json_file), 'r') as fd:
        model_config = json.load(fd)

    # Model source file
    py_files = [f for f in os.listdir(model_dir)
                if os.path.splitext(f)[1] == '.py']
    if len(py_files) != 1:
        raise NameError(f'{model_dir} must contain exactly one .py file.')
    py_file = py_files[0]
    # Import that module
    spec = importlib.util.spec_from_file_location(
        'model_source', os.path.join(model_dir, py_file))
    model_source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_source)
    model_class = getattr(model_source, model_config['module'])

    ''' Segmentation setup '''

    # Initialize the model
    model: nn.Module = model_class(**model_config['init'])
    # Load saved weights
    model.load_state_dict(torch.load(os.path.join(model_dir, pth_file)))

    model.eval()

    # Set up the window config for `prob_maps`
    if window_spacing is None:
        window_spacing = model_config['window']['shape_out']
    window_config = {**model_config['window'], 'spacing': window_spacing}
    n_classes = model_config['init']['n_classes']

    ''' Run segmentation '''

    class_probabilities = prob_maps(
        image_file=image_file,
        model=model,
        model_config=window_config,
        n_classes=n_classes,
        device=device)

    if return_probs:
        if save_file is not None:
            probs_rescaled = (255 * class_probabilities).astype(np.uint8)
            output_shape = np.squeeze(probs_rescaled).shape
            output = np.zeros((*output_shape, 4), dtype=np.uint8)
            output[..., 2] = probs_rescaled
            output[..., 3] = (255 * (probs_rescaled > 0)).astype(np.uint8)

            save_ext = os.path.splitext(save_file)[1].lower()
            if 'tif' in save_ext:
                tif.imsave(save_file, output, compress=7)
            elif 'png' in save_ext:
                plt.imsave(save_file, output)
        return class_probabilities

    segmentation = (class_probabilities > threshold).astype(np.int32)

    ''' Save segmentation, if specified '''

    if save_file is not None:
        if n_classes == 1:
            seg_rescaled = (255 * segmentation).astype(np.uint8)
        else:
            seg_rescaled = (255 / (n_classes - 1) * segmentation).astype(np.uint8)

        save_ext = os.path.splitext(save_file)[1].lower()
        if 'tif' in save_ext:
            output_shape = np.squeeze(seg_rescaled).shape
            output = 255 * np.ones((*output_shape, 3), dtype=np.uint8)
            output[..., 2] = seg_rescaled
            tif.imsave(save_file, output, compress=7)
        elif 'npy' in save_ext:
            cmap = plt.get_cmap(save_cmap)
            output = (255 * cmap(seg_rescaled))[..., :3].astype(np.uint8)
            np.save(save_file, output)
        elif 'png' in save_ext:
            output_shape = np.squeeze(seg_rescaled).shape
            output = np.zeros((*output_shape, 4), dtype=np.uint8)
            output[..., 2] = seg_rescaled
            output[..., 3] = (255 * (seg_rescaled > 0)).astype(np.uint8)
            # cmap = plt.get_cmap(save_cmap)

            # output = (255 * cmap(seg_rescaled))[..., :3].astype(np.uint8)
            plt.imsave(save_file, output)
        else:
            raise ValueError(f'Save extension {save_ext} not recognized.')

    return segmentation


def prob_maps_online(
        image: np.ndarray,
        model: nn.Module,
        model_config: Dict[str, Any],
        n_classes: int,
        device: Optional[str] = None) -> np.ndarray:
    """Generate segmentation class probability maps for an image already loaded
    into memory.

    Args:
        image (np.ndarray): Image to segment.
        model (nn.Module): Segmentation model.
        model_config (Dict[str, Any]): Dict containing window info. Keys:
            'shape_in' (Tuple[int, int, int]): Input window shape.
            'shape_out' (Tuple[int, int, int]): Output window shape.
            'data_scale' (Tuple[int, int, int]): Data zoom along each axis.
            'spacing' (Tuple[int, int, int]): Spacing between consecutive
                windows.
        n_classes (int): Number of segmentation classes
        device (Optional[str]): If provided, specify which PyTorch device to
            use. Default is the first available GPU.

    Returns:
        (np.ndarray): Image class probability map. Take the argmax along axis 0
            to get the final segmentation.

    """
    window_shape_out = model_config['shape_out']

    if image.shape[-1] in [3, 4]:
        image = image[..., 0]
    net_is_2d = model_config['shape_in'][0] == 1
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    shape_out = model_config['shape_out']
    data_scale = model_config['data_scale']

    if net_is_2d:
        resize_shape = [1, 1] + [s // z for s, z in
                                 zip(shape_out[1:], data_scale[1:])]
    else:
        resize_shape = [1, 1] + [s // z for s, z in zip(shape_out, data_scale)]

    prob_shape = [n_classes] + list(image.shape)
    probs = np.zeros(prob_shape, dtype=np.float32)
    pred_count = np.zeros_like(probs, dtype=np.int)

    gen_x, gen_corner = window_batch(image, model_config)
    for i, (x_np, corner_out) in enumerate(zip(gen_x, gen_corner)):
        x_torch = torch.from_numpy(np.expand_dims(x_np, axis=0)).to(device)
        prediction = torch.sigmoid(model(x_torch))
        probs_out = prediction.cpu().detach().numpy()
        probs_out = resize(
            probs_out,
            resize_shape,
            preserve_range=True,
            anti_aliasing=False,
            order=1)
        if not net_is_2d:
            probs_out = np.squeeze(probs_out, axis=1)
        # probs_out = torch.squeeze(prediction.cpu().detach()).numpy()
        # probs_out = np.expand_dims(probs_out, axis=1)

        z0 = corner_out[0]
        z1 = corner_out[0] + window_shape_out[0] // data_scale[0]
        y0 = corner_out[1]
        y1 = corner_out[1] + window_shape_out[1] // data_scale[1]
        x0 = corner_out[2]
        x1 = corner_out[2] + window_shape_out[2] // data_scale[2]
        window_region = \
            (slice(None, None), slice(z0, z1), slice(y0, y1), slice(x0, x1))
        probs[window_region] += probs_out
        pred_count[window_region] += 1

    probs = np.divide(probs, pred_count)

    return probs


def prob_maps(
        image_file: str,
        model: nn.Module,
        model_config: Dict[str, Any],
        n_classes: int,
        device: Optional[str] = None) -> np.ndarray:
    """Generate segmentation class probability maps for a large volume with
    stitched-together windows.

    Args:
        image_file (str): Image file to segment.
        model (nn.Module): Segmentation model.
        model_config (Dict[str, Any]): Dict containing window info. Keys:
            'shape_in' (Tuple[int, int, int]): Input window shape.
            'shape_out' (Tuple[int, int, int]): Output window shape.
            'spacing' (Tuple[int, int, int]): Spacing between consecutive
                windows.
        n_classes (int): Number of segmentation classes
        device (Optional[str]): If provided, specify which PyTorch device to
            use. Default is the first available GPU.

    Returns:
        (np.ndarray): Image class probability map. Take the argmax along axis 0
            to get the final segmentation.

    """
    window_shape_out = model_config['shape_out']

    if isinstance(image_file, str):
        image = b3d.load(
            os.path.dirname(image_file),
            os.path.basename(image_file),
            normalization='white_balance')
    else:
        image = white_balance(image_file)
    if image.shape[-1] in [3, 4]:
        image = image[..., 0]
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    net_is_2d = model_config['shape_in'][0] == 1

    if device is not None:
        device = torch.device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    shape_out = model_config['shape_out']
    data_scale = model_config['data_scale']

    if net_is_2d:
        resize_shape = [1, 1] + [s // z for s, z in
                                 zip(shape_out[1:], data_scale[1:])]
    else:
        resize_shape = [1, 1] + [s // z for s, z in zip(shape_out, data_scale)]

    prob_shape = [n_classes] + list(image.shape)
    probs = np.zeros(prob_shape, dtype=np.float32)
    pred_count = np.zeros_like(probs, dtype=np.int)

    gen_x, gen_corner = window_batch(image, model_config)
    for i, (x_np, corner_out) in enumerate(zip(gen_x, gen_corner)):
        x_torch = torch.from_numpy(np.expand_dims(x_np, axis=0)).to(device)
        prediction = torch.sigmoid(model(x_torch))
        probs_out = prediction.cpu().detach().numpy()
        probs_out = resize(probs_out, resize_shape, preserve_range=True, anti_aliasing=False, order=1)
        if not net_is_2d:
            probs_out = np.squeeze(probs_out, axis=1)
        # probs_out = torch.squeeze(prediction.cpu().detach()).numpy()
        # probs_out = np.expand_dims(probs_out, axis=1)
        z0 = corner_out[0]
        z1 = corner_out[0] + window_shape_out[0] // data_scale[0]
        y0 = corner_out[1]
        y1 = corner_out[1] + window_shape_out[1] // data_scale[1]
        x0 = corner_out[2]
        x1 = corner_out[2] + window_shape_out[2] // data_scale[2]
        window_region = \
            (slice(None, None), slice(z0, z1), slice(y0, y1), slice(x0, x1))
        probs[window_region] += probs_out
        pred_count[window_region] += 1

    probs = np.divide(probs, pred_count)
    # probs = probs / probs.sum(axis=0)[None, ...]

    return probs


def tif_cmap(c) -> np.ndarray:
    """Convert a matplotlib cmap into a tifffile cmap

    Args:
        c: Name of a matplotlib cmap.

    Returns:
        (np.ndarray): tifffile-style cmap.

    """
    colors_8bit = plt.get_cmap(c)(np.arange(256))
    return np.swapaxes(255 * colors_8bit, 0, 1)[0:3, :].astype(np.uint8)


def window_batch(
        image: np.ndarray,
        window_config: Dict[str, Any]) -> Tuple[
            Generator[np.ndarray, None, None],
            Generator[Tuple[int, int, int], None, None]]:
    """Create a batch of image windows that tile an input image.

    Args:
        image (np.ndarray): Source image to break up into windows.
        window_config: Dict containing window structuring info. See
            experiments.membrane.segment.prob_maps docstring for full details.

    Returns:
        (Generator[np.ndarray, None, None]): Numpy input window generator
        (Generator[Tuple[int, int, int], None, None]): Generator of the
            top-left-back corners of each output window within the full image's
            coordinate system.

    """
    scaled_window_shape_in = window_config['shape_in']
    scaled_window_shape_out = window_config['shape_out']
    scaled_window_spacing = window_config['spacing']
    data_scale = window_config['data_scale']

    window_shape_in = [
        rescale(np.zeros(sz), scale=1 / sc).shape[0]
        for sz, sc in zip(scaled_window_shape_in, data_scale)]
    window_shape_out = [
        rescale(np.zeros(sz), scale=1 / sc).shape[0]
        for sz, sc in zip(scaled_window_shape_out, data_scale)]
    window_spacing = [
        rescale(np.zeros(sz), scale=1 / sc).shape[0]
        for sz, sc in zip(scaled_window_spacing, data_scale)]

    spatial_shape = list(image.shape) if len(image.shape) == 3 \
        else [1] + list(image.shape)

    output_corners = gen_corner_points(
        spatial_shape=spatial_shape,
        window_spacing=window_spacing,
        window_shape=window_shape_out,
        random_windowing=False)

    input_corners = gen_conjugate_corners(
        corner_points=output_corners,
        window_shape=window_shape_out,
        conjugate_window_shape=window_shape_in)

    input_window_generator = scalable_window_generator(
        data_volume=image,
        window_shape=window_shape_in,
        scaled_window_shape=scaled_window_shape_in,
        corner_points=input_corners,
        interp_order=0)

    # output_corner_tuples = ((z, y, x)
    #                         for z in output_corners[0]
    #                         for y in output_corners[1]
    #                         for x in output_corners[2])

    output_corner_tuples = ((z, y, x)
                            for y in output_corners[1]
                            for x in output_corners[2]
                            for z in output_corners[0])

    return input_window_generator, output_corner_tuples


if __name__ == '__main__':

    # pths = ['best_weights.pth', 'weights_30.pth', 'weights_15.pth']

    ''' 2D Example '''

    # image_file = '/home/matt/Dropbox/nibib/data/platelet-membrane/sample/train/0010.png'
    # model_dir = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments/isbi2021/output/1010/7'
    # pths = [f for f in os.listdir(model_dir) if 'pth' in os.path.splitext(f)[1]]
    # pth_files = [os.path.join(model_dir, f) for f in pths]
    # save_file = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments/isbi2021/output/1010/7/media/segmentations/ensemble1_z0010_prob.png'
    # ensemble_mode = 'max'
    # os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # save_cmap = 'bone'
    # window_spacing = (1, 80, 80)
    # device = 'cuda:0'
    # threshold = 0.1
    # return_probs = True
    # verbose = True

    ''' 3D Example '''

    # image_file = '/home/matt/Dropbox/nibib/data/platelet-membrane/sample/train/image_10-20.tif'
    # model_dir = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments/isbi2021/output/1016/0'
    # pths = [f for f in os.listdir(model_dir) if 'pth' in os.path.splitext(f)[1]]
    # # pths = ['best_weights.pth', 'weights_5.pth', 'weights_15.pth', 'weights_30.pth']
    # pth_files = [os.path.join(model_dir, f) for f in pths]
    # save_file = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments' \
    #             '/isbi2021/output/1016/0/media/segmentations/ensemble1_t06.tif'
    # ensemble_mode = 'max'
    # os.makedirs(os.path.dirname(save_file), exist_ok=True)
    # save_cmap = 'bone'
    # window_spacing = (1, 100, 100)
    # device = 'cuda:0'
    # threshold = 0.6
    # return_probs = False
    # verbose = True

    ''' 3D Biowulf ensemble example'''

    image_file = load_image_dir(
        '/home/matt/Dropbox/nibib/data/platelet-membrane/sample/eval-far/image/raw')
    # image_file = '/home/matt/Dropbox/nibib/data/platelet-membrane/sample/eval-near/eval-near-filtered.tif'
    root_dir = '/home/matt/experiments/isbi2021/output/biowulf/final/errorblend'
    pth_files = [os.path.join(root_dir, d, 'best_weights.pth')
                 for d in os.listdir(root_dir)
                 if os.path.isdir(os.path.join(root_dir, d))]
    save_file = '/home/matt/Dropbox/nibib/code/leapmanlab/experiments' \
                '/isbi2021/output//biowulf/segmentations/testerrorblend_eval-far.tif'
    ensemble_mode = 'max'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    save_cmap = 'bone'
    window_spacing = (1, 80, 80)
    device = 'cuda:1'
    threshold = 0.5
    return_probs = True
    verbose = True

    segment(
        image_file,
        pth_files[:2],
        save_file,
        ensemble_mode,
        save_cmap,
        window_spacing,
        device,
        threshold,
        return_probs,
        verbose)

    # model_dir = '/home/matt/experiments/membrane/output/0927/4'
    # save_file = '0000_2.png'
    # save_cmap = 'jet'
    # window_spacing = None
    # device = 'cuda:1'
    # threshold = 0.5
    #
    # args = sys.argv[1:]
    # if len(args) > 0:
    #     image_file = args[0]
    # if len(args) > 1:
    #     model_dir = args[1]
    # if len(args) > 2:
    #     save_file = args[2]
    # if len(args) > 3:
    #     save_cmap = args[3]
    # if len(args) > 4:
    #     window_spacing = [int(c) for c in args[4].split(',')]
    # if len(args) > 5:
    #     device = args[5]
    # if len(args) > 6:
    #     threshold = float(args[6])
    #
    # segment(
    #     image_file,
    #     model_dir,
    #     save_file,
    #     save_cmap,
    #     window_spacing,
    #     device,
    #     threshold)
