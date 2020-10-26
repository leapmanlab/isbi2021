"""ISBI 2021 main file. Cell membrane label bootstrapping for SBF-SEM.

"""
import json
import logging
import os
import random
import shutil
import sys
import time

import matplotlib
# matplotlib.use('Agg')

import comet_ml
import leapmanlab as lab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from leapmanlab.create_weight_volume import class_balance_weights
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from src.PlateletIterableDataset import PlateletIterableDataset
# noinspection PyUnresolvedReferences
from src.generate_weights import mistake_correction
# noinspection PyUnresolvedReferences
from src.models.thin3dunet import Thin3DUNet
# noinspection PyUnresolvedReferences
from src.models.unet import UNet
# noinspection PyUnresolvedReferences
from src.segment import segment_online
# noinspection PyUnresolvedReferences
from src.utils import scalable_batch_generator, DuplicateFilter, \
    PlateletDataset, imshow, init_weights, get_n_params, load_image_dir, \
    load_label_dir, white_balance

# Base directory for LCIMB projects
lcimb_dir = os.path.join('..', '..', '..', '..')

# Text file containing Comet.ML API key
comet_key_file = os.path.join(lcimb_dir, 'cometapikey.txt')

# Default experiment parameters. Can be overridden by user-supplied values from
# the commandline or config files
default_params = {
    # Device to use
    'device': 'cuda:1',
    # Directory to save experiment output into
    'experiment_dir': os.path.join('output', time.strftime('%m%d')),
    # Comet.ML experiment name
    'experiment_name': 'isbi2021-scratch',
    # Any additional Comet.ML experiment tags
    'experiment_tags': ['bootstrap-iteration-5'],
    # Image data directory
    'data_dir': os.path.join(lcimb_dir, 'data', 'platelet-membrane', 'sample'),
    # Which directory to use for training data (inside the image data directory)
    'train_dir': 'train-expanded',
    # Which image preprocessing filters to use. Can be 'raw',
    # 'filtered-unsharp', or 'filtered-unsharp-median'
    'preprocessing_filters': 'filtered-unsharp-median',
    # Save trained weights after these epochs (in addition to the system that
    # continually saves the best weights seen so far as judged by eval MIoU)
    'weight_save_epochs': (1, 2, 3, 5, 15, 30),
    # Model architecture settings
    'model_settings': {
        # Model type. Possible types are '2d-unet' and 'thin-3d-unet'
        'model_type': '2d-unet',
        # Number of conv ops per block in the model downsampling path
        'n_convs_per_down_block': 2,
        # Number of conv ops per block in the model upsampling path
        'n_convs_per_up_block': 2,
        # Initial zoom factor for network data processing
        'data_scale': [1, 1, 1],
        # Network data z shape (input, output, and everywhere in between). Only
        # needed for the 'thin-3d-unet' model
        'data_size_z': 3,
        # Network input shape (y+x)
        'data_size_in': 400,
        # Network output shape (y+x)
        'data_size_out': 400,
        # Upsampling mode
        'up_mode': 'upconv',
        # If True, use separable convs
        'separable': False,
        # If True, use LeakyReLUs in place of ReLUs
        'leaky': True,
        # If True, use instance normalization
        'instance_norm': True,
        # U-Net depth, i.e. number of pooling or unpooling layers minus one
        'unet_depth': 4,
        # Number of initial conv layer filters
        'n_init_filters': 32,
        # Network conv padding
        'padding': True,
        # Threshold for positive prediction
        'detect_threshold': 0.1

    },
    # Model training settings
    'train_settings': {
        # Space between consecutive windows passed through the network during
        # training (z axis)
        'window_spacing_z': 1,
        # Space between consecutive windows passed through the network during
        # training (y+x axes only)
        'window_spacing': 16,
        # Number of slices to use for training. -1 for all slices
        'n_train_slices': 1,
        # Saved weight file to load before beginning training, if provided
        'pretrained_model_pth': None,
        # If >0, add Gaussian noise to the pretrained weights during
        # initialization. For each param tensor, scale the noise's std to
        # be `pretrained_noise_std_scale` times the param's std
        'pretrained_noise_std_scale': 0,
        # Floor on the error weighting array
        'weight_floor': 0.08,
        # Scaling hyperparameter for the prediction error weight term in the
        # training error weight
        'pred_error_weight_scale': 100,
        # Prediction error mode. Currently 'positive', 'negative', or 'both'
        # are supported. The 'positive' mode penalizes only false positives,
        # 'negative' penalizes only false negatives, and 'both' penalizes both
        'pred_error_mode': 'negative',
        # Standard deviation of the Gaussian blurring kernel applied to
        # prediction error weighting
        'pred_error_blur_std': 5,
        # Minimum size of a predicted positive region to be included in
        # prediction error weighting
        'pred_error_min_size': 10,
        # If True, blend BI 4 prediction error with pre-BI 4 prediction errors
        'pred_error_blending': False,
        # Number of training epochs
        'n_epochs': 100,
        # Minibatch size
        'batch_size': 1,
        # Learning rate
        'learning_rate': 10**-2.8,
        # Weight decay
        'weight_decay': 1e-5,
        # Adam epsilon parameter
        'adam_epsilon': 0.001,
        # Learning rate scheduler settings
        'lr_scheduler_settings': {
            # Scaling factor after performance plateaus
            'factor': 0.5,
            # Plateau patience before scaling
            'patience': 5,
            # Cooldown after scaling before tracking performance again
            'cooldown': 0,
            # Minimum learning rate
            'min_lr': 1e-6
        },
        # Print a report during training?
        'show_train_report': True,
        # Number of training iterations between printed update reports
        'its_per_report': 500,
        # Display predictions during training?
        'show_train_image': False,
        # Number of training iterations between prediction images
        'its_per_image': 10
    },
    # Data deformation settings.
    'deformation_settings': {
        # Deformation spatial scale. Larger value = higher-frequency
        # deformations
        'scale': 80,
        # Deformation amplitude mean. Larger value = larger deformations
        'alpha': 12,
        # Deformation amplitude standard deviation. Larger value = more variance
        # in deformation strength across the image
        'sigma': 0.7
    },
    # RNG seeds for random processes in training
    'rng_seeds': {
        # Base seed for PyTorch RNG for random weight initialization
        'model': random.randint(0, 2**32),
        # Base seed for data deformation and presentation order
        'data': random.randint(0, 2**32)},
}


def main():
    plt.ion()
    params = lab.experiment_params(default_params)
    device = params['device']
    experiment_dir = params['experiment_dir']
    experiment_name = params['experiment_name']
    experiment_tags = params['experiment_tags']
    data_dir = params['data_dir']
    train_dir = params['train_dir']
    preprocessing_filters = params['preprocessing_filters']
    weight_save_epochs = params['weight_save_epochs']
    model_settings = params['model_settings']
    train_settings = params['train_settings']
    deformation_settings = params['deformation_settings']
    rng_seeds = params['rng_seeds']

    model_type = model_settings['model_type']
    n_convs_per_down_block = model_settings['n_convs_per_down_block']
    n_convs_per_up_block = model_settings['n_convs_per_up_block']
    data_scale = model_settings['data_scale']
    data_size_z = model_settings['data_size_z']
    data_size_in = model_settings['data_size_in']
    data_size_out = model_settings['data_size_out']
    up_mode = model_settings['up_mode']
    separable = model_settings['separable']
    leaky = model_settings['leaky']
    instance_norm = model_settings['instance_norm']
    unet_depth = model_settings['unet_depth']
    n_init_filters = model_settings['n_init_filters']
    padding = model_settings['padding']
    detect_threshold = model_settings['detect_threshold']

    window_spacing_z = train_settings['window_spacing_z']
    window_spacing = train_settings['window_spacing']
    n_train_slices = train_settings['n_train_slices']
    pretrained_model_pth = train_settings['pretrained_model_pth']
    pretrained_noise_std_scale = train_settings['pretrained_noise_std_scale']
    weight_floor = train_settings['weight_floor']
    pred_error_weight_scale = train_settings['pred_error_weight_scale']
    pred_error_mode = train_settings['pred_error_mode']
    pred_error_blur_std = train_settings['pred_error_blur_std']
    pred_error_min_size = train_settings['pred_error_min_size']
    pred_error_blending = train_settings['pred_error_blending']
    n_epochs = train_settings['n_epochs']
    batch_size = train_settings['batch_size']
    learning_rate = train_settings['learning_rate']
    weight_decay = train_settings['weight_decay']
    adam_epsilon = train_settings['adam_epsilon']
    lr_factor = train_settings['lr_scheduler_settings']['factor']
    lr_patience = train_settings['lr_scheduler_settings']['patience']
    lr_cooldown = train_settings['lr_scheduler_settings']['cooldown']
    lr_min_lr = train_settings['lr_scheduler_settings']['min_lr']
    show_train_report = train_settings['show_train_report']
    its_per_report = train_settings['its_per_report']
    show_train_image = train_settings['show_train_image']
    its_per_image = train_settings['its_per_image']

    assert model_type in ('2d-unet', 'thin-3d-unet')

    # Logger setup
    filt = DuplicateFilter()
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger = logging.getLogger('Logger')
    logger.addHandler(stdout_handler)
    logger.addFilter(filt)
    logger.setLevel(logging.DEBUG)

    # Comet.ML experiment setup
    with open(comet_key_file, 'r') as fd:
        api_key = fd.read().strip()
    experiment = comet_ml.Experiment(
        api_key=api_key,
        project_name=experiment_name,
        workspace='lcimb',
        auto_metric_logging=True)
    experiment.log_parameters(params)
    for tag in experiment_tags:
        experiment.add_tag(tag)
    experiment.add_tag(model_type)

    # Output directory setup
    output_dir = lab.create_output_dir(experiment_dir)
    media_dir = os.path.join(output_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    lab.save_config(os.path.join(output_dir, 'experiment.cfg'), params)
    lab.archive_experiment(
        os.path.dirname(os.path.realpath(__file__)),
        output_dir,
        ['py'])
    # Tag the experiment with the output folder name
    dir_parts = [p for p in output_dir.split('/') if p]
    experiment.add_tag(f'{"/".join(dir_parts[-2:])}')

    # Add a filehandler to the logger to save a log in the output dir
    log_file = os.path.join(output_dir, 'log.log')
    logger.addHandler(logging.FileHandler(log_file))

    # 2D vs 3D model settings. For the 2D model, ignore the specified data
    # z size and force it to use z size 1

    if model_type == '2d-unet':
        data_size_z = 1
        window_spacing_z = 1

    # Data windowing configuration for training and eval datasets
    train_windowing_params = {
        'scaled_image_window_shape':
            [data_size_z, data_size_in, data_size_in],
        'scaled_label_window_shape':
            [data_size_z, data_size_out, data_size_out],
        'scaled_window_spacing':
            [window_spacing_z, window_spacing, window_spacing],
        'random_windowing': True}
    eval_windowing_params = {
        'scaled_image_window_shape':
            [data_size_z, data_size_in, data_size_in],
        'scaled_label_window_shape':
            [data_size_z, data_size_out, data_size_out],
        'scaled_window_spacing':
            [data_size_z, data_size_out, data_size_out],
        'random_windowing': False}
    experiment.log_parameters(train_windowing_params)
    experiment.log_parameters(eval_windowing_params)

    # Inference data windowing configuration
    forward_windowing_params = {
        'shape_in': [data_size_z, data_size_in, data_size_in],
        'shape_out': [data_size_z, data_size_out, data_size_out],
        'data_scale': data_scale}
    experiment.log_parameters(forward_windowing_params)

    # Load the data

    train_image_dir = os.path.join(data_dir, train_dir, 'image',
                                   preprocessing_filters)

    train_image = load_image_dir(train_image_dir, n_train_slices)
    n_train_slices = train_image.shape[0]

    experiment.add_tag(f'{n_train_slices}-slices')

    train_label_dir = os.path.join(data_dir, train_dir, 'label', 'corrected')
    train_label = load_label_dir(train_label_dir, n_train_slices)

    if pred_error_blending:
        pred_error_weight_slices = \
            [pred_error_weight_scale *
             mistake_correction(i, 0, pred_error_mode,
                                pred_error_min_size,
                                os.path.join(data_dir, train_dir),
                                os.path.join('label', 'raw'),
                                os.path.join('label', 'corrected'))
             for i in range(n_train_slices)]
        pred_error_weight_stack = np.stack(pred_error_weight_slices, axis=0)

        # Load the prediction error for the 10x800x800 BI 3 training region and
        # use it instead of the BI 4 prediction errors
        old_train_dir = os.path.join(data_dir, 'train')
        old_pred_weight_slices = \
            [pred_error_weight_scale *
             mistake_correction(i, 0, pred_error_mode, pred_error_min_size, old_train_dir, 'label-raw', 'label')
             for i in range(1, 10)]
        old_pred_weight_stack = np.stack(old_pred_weight_slices, axis=0)
        pred_error_weight_stack[3:12, 200:1000, 200:1000] = np.maximum(
            old_pred_weight_stack,
            pred_error_weight_stack[3:12, 200:1000, 200:1000])
        pred_error_weight_stack = gaussian_filter(
            pred_error_weight_stack.astype(float),
            sigma=pred_error_blur_std)
        pred_error_weight_stack = pred_error_weight_stack / \
                                  pred_error_weight_stack.max()
    else:
        pred_error_weight_slices = \
            [pred_error_weight_scale *
             mistake_correction(i, pred_error_blur_std, pred_error_mode,
                                pred_error_min_size,
                                os.path.join(data_dir, train_dir),
                                os.path.join('label', 'raw'),
                                os.path.join('label', 'corrected'))
             for i in range(n_train_slices)]
        pred_error_weight_stack = np.stack(pred_error_weight_slices, axis=0)
    pred_error_weight = 1 + pred_error_weight_stack

    weight_sample = pred_error_weight[0]
    weight_sample = (weight_sample - weight_sample.min()) / \
                    (weight_sample.max() - weight_sample.min())
    error_viz = np.zeros(list(weight_sample.shape) + [4])
    error_viz[..., 0] = weight_sample
    error_viz[..., 3] = weight_sample**(1/3)
    experiment.log_image(error_viz, name='pred_error_weight sample')

    # Generate class frequency balance weights
    balance_weight = class_balance_weights(train_label)

    # Train weight multiplies class balancing and prediction error weighting
    train_weight = np.maximum(weight_floor, pred_error_weight * balance_weight)

    # Eval data setup

    eval_near_image_dir = os.path.join(data_dir, 'eval-near', 'image',
                                       preprocessing_filters)
    eval_near_image = load_image_dir(eval_near_image_dir, -1)
    eval_near_label_dir = os.path.join(data_dir, 'eval-near', 'label', 
                                       'corrected')
    eval_near_label = load_label_dir(eval_near_label_dir, -1)

    eval_far_image_dir = os.path.join(data_dir, 'eval-far', 'image',
                                       preprocessing_filters)
    eval_far_image = load_image_dir(eval_far_image_dir, -1)
    eval_far_label_dir = os.path.join(data_dir, 'eval-far', 'label',
                                       'corrected')
    eval_far_label = load_label_dir(eval_far_label_dir, -1)

    if model_type == '2d-unet':
        # Build the model
        model = UNet(
            n_classes=1,
            padding=padding,
            up_mode=up_mode,
            depth=unet_depth,
            n_init_filters=n_init_filters,
            instance_norm=instance_norm,
            separable=separable,
            leaky=leaky)
    elif model_type == 'thin-3d-unet':
        model = Thin3DUNet(
            z_size=data_size_z,
            n_classes=1,
            n_convs_per_down_block=n_convs_per_down_block,
            n_convs_per_up_block=n_convs_per_up_block,
            depth=unet_depth,
            n_init_filters=n_init_filters,
            padding=padding,
            instance_norm=instance_norm,
            up_mode=up_mode,
            separable=separable,
            leaky=leaky)
    else:
        raise ValueError(f'Model type {model_type} not recognized')

    # Load pretrained weights if specified
    if pretrained_model_pth is not None:
        model.load_state_dict(torch.load(pretrained_model_pth))
        # If the std scaling > 0, add Gaussian noise to the weights
        if pretrained_noise_std_scale > 0:
            with torch.no_grad():
                for param in model.parameters():
                    std = torch.std(param).item()
                    if not np.isnan(std):
                        param.add_(
                            torch.randn(param.size())
                            * pretrained_noise_std_scale * std)

    # Create a model settings JSON file to save along with trained weights
    # Currently hardwired to save UNet config info and source file
    if model_type == '2d-unet':
        model_json = {
            'module': 'UNet',
            'init': {
                'in_channels': 1,
                'n_classes': 1,
                'up_mode': up_mode,
                'depth': unet_depth,
                'n_init_filters': n_init_filters,
                'padding': padding,
                'instance_norm': instance_norm,
                'separable': separable,
                'leaky': leaky
            },
            'window': forward_windowing_params
        }
    elif model_type == 'thin-3d-unet':
        model_json = {
            'module': 'Thin3DUNet',
            'init': {
                'z_size': data_size_z,
                'in_channels': 1,
                'n_classes': 1,
                'n_convs_per_down_block': n_convs_per_down_block,
                'n_convs_per_up_block': n_convs_per_up_block,
                'depth': unet_depth,
                'n_init_filters': n_init_filters,
                'padding': padding,
                'instance_norm': instance_norm,
                'up_mode': up_mode,
                'separable': separable,
                'leaky': leaky
            },
            'window': forward_windowing_params
        }
    else:
        raise ValueError(f'Model type {model_type} not recognized')

    with open(os.path.join(output_dir, 'model.json'), 'w') as fd:
        json.dump(model_json, fd)
    # Save the UNet src file to the output directory
    isbi2021_src_dir = os.path.dirname(os.path.realpath(__file__))

    if model_type == '2d-unet':
        src_name = 'unet.py'
    elif model_type == 'thin-3d-unet':
        src_name = 'thin3dunet.py'
    else:
        raise ValueError(f'Model type {model_type} not recognized')

    unet_src_file = os.path.join(isbi2021_src_dir, 'src', 'models', src_name)
    shutil.copyfile(unet_src_file, os.path.join(output_dir, src_name))

    # Move the model to device
    model.to(device)

    # Set random seeds
    # Seed before model weight init, then again before DataLoader shuffling
    np.random.seed((rng_seeds['model'] + 3) % 2 ** 32)
    random.seed((rng_seeds['model'] + 2) % 2 ** 32)
    torch.manual_seed((rng_seeds['model'] + 1) % 2 ** 32)
    torch.cuda.manual_seed_all((rng_seeds['model']) % 2 ** 32)

    # Initialize weights if necessary
    if pretrained_model_pth is None:
        model.apply(init_weights)

    # Optimization setup

    optim = torch.optim.Adam(
        model.parameters(),
        weight_decay=weight_decay,
        lr=learning_rate,
        eps=adam_epsilon)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        factor=lr_factor,
        patience=lr_patience,
        verbose=True,
        cooldown=lr_cooldown,
        min_lr=lr_min_lr)

    # Reseed all the torch-related RNGs for deterministic data shuffling
    np.random.seed((rng_seeds['data'] + 3) % 2 ** 32)
    random.seed((rng_seeds['data'] + 2) % 2 ** 32)
    torch.manual_seed((rng_seeds['data'] + 1) % 2 ** 32)
    torch.cuda.manual_seed_all((rng_seeds['data']) % 2 ** 32)

    # Create a fixed eval batch
    # eval_images, eval_labels = scalable_batch_generator(
    #     eval_image,
    #     eval_label,
    #     data_scale,
    #     return_generators=True,
    #     **eval_windowing_params)
    # eval_dataset = PlateletIterableDataset(
    #     eval_images,
    #     eval_labels,
    #     train=False)
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=1)

    # Color settings for plots generated during training and eval
    plot_settings = (
        {'cmap': 'gray'},
        {'cmap': 'jet', 'vmin': 0, 'vmax': train_label.max()},
        {'cmap': 'jet', 'vmin': 0, 'vmax': train_label.max()})

    if show_train_image:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
        obj0 = None
        obj1 = None

    # Track best eval MIoU for model saving
    best_miou = 0
    for epoch in range(n_epochs):
        time0 = time.time()
        experiment.set_epoch(epoch)
        with experiment.train():
            deformation_settings['seed'] = (rng_seeds['data'] + epoch) % 2 ** 32
            images, labels, weights = scalable_batch_generator(
                image=train_image,
                label=train_label,
                data_scale=data_scale,
                weight=train_weight,
                do_deformation=True,
                deformation_settings=deformation_settings,
                return_generators=True,
                **train_windowing_params)

            train_dataset = PlateletIterableDataset(
                images,
                labels,
                weights,
                train=True)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=1)
            model.train()

            n_examples = 0

            for i, (x, y, w) in enumerate(train_dataloader):
                if show_train_image and i % its_per_image == 0:
                    x_img = np.squeeze(x.detach().numpy())[0]
                    cmap_gray = plt.get_cmap('gray')
                    x_img_rgb = cmap_gray(x_img)
                    if i == 0:
                        obj0 = axs[0].imshow(x_img_rgb)
                    else:
                        obj0.set_data(x_img_rgb)
                    axs[0].set_title(f'Train image {i}')
                assert len(torch.unique(y)) == 2
                n_examples += 1
                x = x.to(device)
                y = y.float().to(device)
                w = torch.squeeze(w.to(device))
                prediction = torch.squeeze(model(x), dim=1)
                if show_train_image and i % its_per_image == 0:
                    pred_img = np.squeeze(prediction.cpu().detach().numpy())[0]
                    cmap_pred = plt.get_cmap('Blues')
                    pred_img_rgb = cmap_pred(pred_img)
                    overlay_rgb = np.copy(x_img_rgb)
                    overlay_rgb[pred_img > 0.01, :] = pred_img_rgb[pred_img > 0.01, :]
                    if i == 0:
                        obj1 = axs[1].imshow(overlay_rgb)
                    else:
                        obj1.set_data(overlay_rgb)
                    axs[1].set_title(f'Prediction overlay {i}')
                    plt.show()
                    plt.pause(0.0001)
                loss = F.binary_cross_entropy_with_logits(
                    prediction,
                    torch.squeeze(y, dim=1),
                    reduction='none')
                loss = torch.sum(torch.mul(loss, w))

                optim.zero_grad()
                loss.backward()
                optim.step()
                experiment.log_metric('loss', loss.item())
                if show_train_report and (i + 1) % its_per_report == 0:
                    time1 = time.time()
                    dtime = time1 - time0
                    batch_per_sec = its_per_report / dtime
                    its_per_sec = batch_per_sec * batch_size
                    time0 = time1

                    print(
                        f'Epoch [{epoch + 1}/{n_epochs}], Step {i + 1} '
                        f'({batch_per_sec:.2g} batch/sec, {its_per_sec:.2g} '
                        f'its/sec), Loss: {loss.item():.3f}')

                    classes = (prediction > detect_threshold).int()

                    if data_size_z > 1:
                        z_half = (data_size_z - 1) // 2
                        x = x[..., z_half, :, :]
                        classes = classes[..., z_half, :, :]
                        y = y[..., z_half, :, :]

                    if batch_size > 1:

                        images = (np.squeeze(x.cpu().numpy())[0],
                                  np.squeeze(classes.cpu().numpy())[0],
                                  np.squeeze(y.cpu().numpy())[0])
                    else:
                        images = (np.squeeze(x.cpu().numpy()),
                                  np.squeeze(classes.cpu().numpy()),
                                  np.squeeze(y.cpu().numpy()))

                    fig = imshow(images, (15, 5), plot_settings)
                    experiment.log_figure(
                        figure_name=f"epoch_{epoch + 1}_step_{i + 1}",
                        figure=fig)
                    plt.close(fig)

            print(f'Epoch {epoch}: {n_examples} samples')

        # Get stats on eval data
        eval_images = [eval_near_image, eval_far_image]
        eval_labels = [eval_near_label, eval_far_label]
        eval_names = ['Near', 'Far']
        for eval_image, eval_label, eval_name in \
                zip(eval_images, eval_labels, eval_names):
            if eval_image.shape[0] > 1:
                save_file = os.path.join(media_dir, f'epoch_{epoch:04}.tif')
            else:
                save_file = os.path.join(media_dir, f'epoch_{epoch:04}.png')
            print(f'\n==========')
            print(f'Eval Stats {eval_name}, Epoch {epoch}:')
            eval_segmentation = segment_online(
                eval_image,
                model,
                forward_windowing_params,
                save_file,
                window_spacing=None,
                device=device)

            z_half = eval_segmentation.shape[0] // 2

            eval_seg_img = np.squeeze(eval_segmentation).astype(float)
            if eval_seg_img.ndim == 3:
                eval_seg_img = eval_seg_img[z_half]
            with experiment.validate():
                experiment.log_image(
                    eval_seg_img, name=f'Eval {eval_name} {epoch}',
                    image_colormap='jet')

                # Compute eval MIoU
                miou = jaccard_score(
                    eval_label.flatten(),
                    eval_segmentation.flatten(),
                    average='macro')

                print(f'  Mean IoU ({eval_name}): {miou}')
                experiment.log_metric(f'eval_{eval_name.lower()}_miou', miou)

                # Compute false positive and false negative rates, using several
                # size thresholds for detection regions

                false_negatives = eval_label.astype(bool) & ~eval_segmentation
                false_positives = ~eval_label.astype(bool) & eval_segmentation

                neg_percs = []
                pos_percs = []

                thresholds = [0, 5, 65, 401]
                for threshold in thresholds:
                    if threshold == 0:
                        thresholded_fn = false_negatives
                        thresholded_fp = false_positives
                    else:
                        thresholded_fn = remove_small_objects(
                            false_negatives, threshold)
                        thresholded_fp = remove_small_objects(
                            false_positives, threshold)

                    neg_perc = thresholded_fn.sum() / thresholded_fn.size * 100
                    neg_percs.append(neg_perc)
                    pos_perc = thresholded_fp.sum() / thresholded_fp.size * 100
                    pos_percs.append(pos_perc)

                    experiment.log_metric(f'eval_{eval_name.lower()}_fn-{threshold}', neg_perc)
                    experiment.log_metric(f'eval_{eval_name.lower()}_fp-{threshold}', pos_perc)

                neg_str = '.  '.join([f'{p:.3f}% ({t})'
                                      for p, t in zip(neg_percs, thresholds)])
                pos_str = '.  '.join([f'{p:.3f}% ({t})'
                                      for p, t in zip(pos_percs, thresholds)])

                print(f'  False negatives (min size): ' + neg_str)
                print(f'  False positives (min size): ' + pos_str)
                print('==========\n')

                if eval_name == 'Near':
                    if miou > best_miou:
                        best_weight_path = os.path.join(output_dir,
                                                        'best_weights.pth')
                        print("Saving best model")
                        best_miou = miou
                        torch.save(
                            model.state_dict(), best_weight_path)

        # Save trained weights after specified epochs
        if epoch in weight_save_epochs:
            weight_save_path = os.path.join(
                output_dir,
                f'weights_{epoch}.pth')
            torch.save(
                model.state_dict(), weight_save_path)

        lr_scheduler.step(loss)

    experiment.end()
    return model


if __name__ == '__main__':
    main()
