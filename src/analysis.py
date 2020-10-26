"""Perform analysis for [Paper 1 title] for a network or ensemble.

"""
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

import genenet as gn
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import pdflatex
#import seaborn as sn
import sklearn
import tensorflow as tf
import tifffile as tif
import bio3d_vision as b3d

from functools import lru_cache

from scipy.ndimage.morphology import distance_transform_edt
from sklearn.utils.multiclass import unique_labels

import leapmanlab as lab

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# Base directory for the lab
lcimb_dir = os.path.join('..', '..', '..', '..')

default_params = {
    # Experiment save directory for the created GeneNets
    'experiment_dir': os.path.join('output', time.strftime('%m%d')),
    # Directory containing the models evaluated in the paper
    # 'model_dir': os.path.join(lcimb_dir, 'output', 'aug2019'),
    # 'model_dir': os.path.join('..', 'unet_3d_1xkxk', 'output', '0930'),
    'model_dir': '/home/matthew/Desktop/remake3d/unet_3d_1xkxk_Sep302019_152446/output/1125',
    # Directory containing eval data
    'eval_data_dir': os.path.join(lcimb_dir, 'data', 'platelet'),
    # Directory containing test data
    'test_data_dir': os.path.join(lcimb_dir, 'data', 'platelet-donor2'),
    # Analyze these networks and ensembles, within the model dir
    # TODO: Revert
    # 'net_dirs_to_analyze': {**{f'2D Net {i}': os.path.join('2d', f'{i}')
    #                         for i in range(2)},
    #                         **{f'2D Ensemble {i}': [os.path.join('2d', f'{n}')
    #                                                 for n in range(i)]
    #                            for i in range(3, 4)}},
    # 'net_dirs_to_analyze': {'3D Net 0': os.path.join('3d', '0')},
    'net_dirs_to_analyze': {'2D-3D 1125': '0'},
    # Test data has 10 possible regions that can be used. Specify which ones to
    # run inference on
    'test_data_vol_idxs': [1],  # [1, 3, 6, 7],
    # Indices of test volumes to display in the stat table
    'stat_table_test_idxs': [1],  # [1, 3, 6, 7],
    # Name of the eval statistic to display in the table. Currently supports
    # 'miou' (mean intersection-over-union) and 'ari' (adjusted Rand index)
    'eval_stat_name': 'miou'
}


def main():
    params = lab.experiment_params(default_params, required_keys=[])

    experiment_dir = params['experiment_dir']
    # Create a directory for this experiment trial's nets
    os.makedirs(experiment_dir, exist_ok=True)

    # output_dir = 'output/1011/2'
    # Create net output dir inside the trial dir
    output_dir = lab.create_output_dir(experiment_dir)
    # Save the experiment params as a config file
    lab.save_config(os.path.join(output_dir, 'experiment.cfg'), params)
    # Archive a copy of this experiment
    lab.archive_experiment(os.path.dirname(os.path.realpath(__file__)),
                           output_dir,
                           ['py'])

    net_output_dir = os.path.join(output_dir, 'nets')
    run_analysis(params, net_output_dir)

    figure_output_dir = os.path.join(output_dir, 'figures')
    build_subfigures(params, figure_output_dir, net_output_dir)
    pass


def build_subfigures(params, figure_dir, net_dir):
    """Create a table of performance stats, a montage of full or restricted
    confusion matrices, and montages of eval and test dataset performance.

    Args:
        params:
        output_dir:

    Returns: None

    """
    stat_dir = figure_dir
    build_stat_table(params, stat_dir, net_dir)
    pass


def build_stat_table(params, output_dir, net_dir):
    """Build a table of statistics for the architectures being compared, save
    as a PDF.

    Args:
        params:
        output_dir:

    Returns:

    """
    os.makedirs(output_dir, exist_ok=True)
    eval_stat_name = params['eval_stat_name']
    stat_table_test_idxs = params['stat_table_test_idxs']
    net_dirs_to_analyze = params['net_dirs_to_analyze']

    header = ['Architecture', f'{eval_stat_name.upper()} Eval', 'MOA Eval'] + \
        [f'MOA Test {i}' for i in range(len(stat_table_test_idxs))]

    ncols = len(header)

    table_rows = [header]

    for key, value in net_dirs_to_analyze.items():
        row = []

        arch_name = key
        row.append(arch_name)

        net_save_dir = os.path.join(net_dir, arch_name)
        eval_dataset = os.path.join(net_save_dir, 'eval')
        test_datasets = [os.path.join(net_save_dir, f'test_{i}')
                         for i in stat_table_test_idxs]

        with open(os.path.join(eval_dataset, 'stats.json'), 'r') as fd:
            eval_stats = json.load(fd)
        eval_stat = eval_stats[eval_stat_name]
        row.append(eval_stat)
        eval_moa = eval_stats['avg_organelle_accuracy']
        row.append(eval_moa)

        for test_dataset in test_datasets:
            with open(os.path.join(test_dataset, 'stats.json'), 'r') as fd:
                test_stats = json.load(fd)
                test_moa = test_stats['avg_organelle_accuracy']
                row.append(test_moa)

        row = [row[0]] + [f'{r:.3f}' for r in row[1:]]
        table_rows.append(row)

    # Bold best results in columns 1+
    for i in range(1, ncols):
        results = [float(r[i]) for r in table_rows[1:]]
        best_idx = int(np.argmax(results))
        best_val = table_rows[best_idx + 1][i]
        table_rows[best_idx + 1][i] = rf'\textbf{{{best_val}}}'

    # Build the Table .tex file
    head = r'''\documentclass{article}
\usepackage{amsmath}
\usepackage{amsthm}
\usepackage{amsfonts}
\usepackage{amssymb}

\makeatletter
\def\thickhline{%
  \noalign{\ifnum0=`}\fi\hrule \@height \thickarrayrulewidth \futurelet
   \reserved@a\@xthickhline}
\def\@xthickhline{\ifx\reserved@a\thickhline
               \vskip\doublerulesep
               \vskip-\thickarrayrulewidth
             \fi
      \ifnum0=`{\fi}}
\makeatother

\newlength{\thickarrayrulewidth}
\setlength{\thickarrayrulewidth}{2\arrayrulewidth}

\begin{document}

\bgroup
\def\arraystretch{1.2}'''
    foot = r'''\thickhline
\end{tabular}
\egroup

\end{document}'''
    ls = ' '.join(['l']*ncols)
    table_start = [rf'\begin{{tabular}}{{{ls}}}', r'\thickhline']
    table_body = [r' & '.join(row) + r' \\' for row in table_rows]
    table_body.insert(1, r'\hline')
    table = '\n'.join(table_start + table_body)

    tex_source = '\n'.join([head, table, foot])
    tex_file = os.path.join(output_dir, 'stat_table.tex')
    with open(tex_file, 'w') as fd:
        fd.write(tex_source)
    pdf_builder = pdflatex.PDFLaTeX.from_texfile(tex_file)
    pdf_builder.set_output_directory(output_dir)
    pdf_builder.create_pdf(keep_pdf_file=True, keep_log_file=False)
    pass


def run_analysis(params, output_dir):
    """Outline of the whole analysis:

    Evaluate several different nets and ensembles.
    For each net/ensemble:
        - Image of segmentation on eval and test data.
        - Jaccard scores and ARI on eval data, all classes
        - Organelle restricted confusion matrix + accuracy average on eval data
        and test data.
            - Meaning: Take the confusion matrix of the samples contained
            within the region where the ground truth labels are nonzero.
            Report just the organelle rows. Average the accuracy scores
            across organelles to get a test data number to compare with eval
            data.

    """
    model_dir = params['model_dir']
    eval_data_dir = params['eval_data_dir']
    test_data_dir = params['test_data_dir']
    net_dirs_to_analyze = params['net_dirs_to_analyze']
    test_data_vol_idxs = params['test_data_vol_idxs']

    for key in net_dirs_to_analyze:
        value = net_dirs_to_analyze[key]
        if isinstance(value, str):
            net_dirs = os.path.join(model_dir, value)
        else:
            net_dirs = [os.path.join(model_dir, d) for d in value]
        # Create a save directory for all the output of `net_dirs`' analysis.

        net_save_dir = os.path.join(output_dir, key)

        # Evaluate on eval data

        eval_image_file = os.path.join(eval_data_dir, 'eval-volume.tif')
        eval_label_file = os.path.join(eval_data_dir, 'eval-label-volume.tif')

        eval_save_dir = os.path.join(net_save_dir, 'eval')
        os.makedirs(eval_save_dir, exist_ok=True)

        print('Starting eval analysis')

        segment_and_evaluate(
            net_dirs,
            eval_image_file,
            eval_label_file,
            eval_save_dir)

        # Evaluate on test data, masked to only look at non-background ground
        # truth labels

        test_image_files = [os.path.join(test_data_dir, f'{j}', 'raw.tif')
                            for j in test_data_vol_idxs]
        test_label_files = [os.path.join(test_data_dir, f'{j}', 'labels.tif')
                            for j in test_data_vol_idxs]
        test_save_dirs = [os.path.join(net_save_dir, f'test_{j}')
                          for j in test_data_vol_idxs]

        i = 0
        for timage, tlabel, tsd in zip(test_image_files, test_label_files,
                                       test_save_dirs):
            os.makedirs(tsd, exist_ok=True)

            print(f'Starting test {i}')
            i += 1

            segment_and_evaluate(net_dirs, timage, tlabel, tsd, do_nonzero_masking=True)

    pass


def evaluate(
        true: np.ndarray,
        prediction: np.ndarray,
        do_nonzero_masking: bool = False,
        labels: list[int] = range(7)) -> Dict[str, Any]:
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
    average_organelle_accuracy = 0.2 * float(normalized_confusion_matrix[2, 2] +
        normalized_confusion_matrix[3, 3] + normalized_confusion_matrix[4, 4]
        + normalized_confusion_matrix[5, 5] + normalized_confusion_matrix[6, 6])
    normalized_confusion_matrix = normalized_confusion_matrix.tolist()

    results = {
        'classes_present': unique_labels(true, prediction).tolist(),
        'miou': miou,
        'ari': ari,
        'confusion_matrix': normalized_confusion_matrix,
        'avg_organelle_accuracy': average_organelle_accuracy}
    return results


def segment_and_evaluate(
        net_dirs: Union[str, List[str]],
        image_file: str,
        label_file: str,
        save_dir: str,
        do_nonzero_masking: bool = False,
        device: Optional[str] = None):
    """

    Args:
        net_dirs:
        image_file:
        label_file:
        save_dir:
        do_nonzero_masking:
        device:

    Returns: None

    """
    if isinstance(net_dirs, str):
        net_dirs = [net_dirs]

    eval_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    semantic_label_file = label_file

    eval_image = b3d.load(eval_dir, image_name)
    semantic_label = b3d.load(eval_dir,
                              semantic_label_file,
                              data_type=np.int32)

    eval_image = (eval_image - eval_image.min()) * 255 / (
                eval_image.max() - eval_image.min())

    segmentation = segment_better(
        net_dirs,
        image_file,
        label_file,
        save_dir,
        device)

    results = evaluate(semantic_label,
                       segmentation,
                       do_nonzero_masking)
    # Append 'net_dirs' to results
    results['net_dirs'] = net_dirs

    # Save result dict as JSON

    result_file = os.path.join(save_dir, 'stats.json')
    with open(result_file, 'w') as fd:
        json.dump(results, fd)

    # Save an image of the confusion matrix

    confusion_matrix_image_file = os.path.join(save_dir, 'confusion_matrix.pdf')
    save_confusion_matrix(
        results['confusion_matrix'],
        save_file=confusion_matrix_image_file,
        class_names=['Background', 'Cell', 'Mito', 'Alpha', 'Canalicular',
                     'Dense', 'Dense Core'],
        classes_present=results['classes_present'],
        cmap=plt.cm.Blues)

    pass


def save_confusion_matrix(
        cm: np.ndarray,
        save_file: str,
        class_names: List[str],
        classes_present: Optional[List[int]] = None,
        title: Optional[str] = None,
        cmap=plt.cm.Blues):
    """

    Args:
        cm:
        save_file:
        class_names:
        classes_present:
        title:
        cmap:

    Returns: None

    """
    if not title:
        title = 'Normalized confusion matrix'

    if not isinstance(class_names, np.ndarray):
        class_names = np.array(class_names)

    if classes_present:
        classes = class_names[classes_present]
    else:
        classes = class_names

    cm = np.array(cm)

    plt.figure()

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.savefig(save_file)
    plt.close(fig)
    pass


def segment_better(
        net_dirs: Union[str, List[str]],
        image_file: str,
        label_file: str,
        save_dir: str,
        device: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """

    Args:
        net_dirs:
        image_file:
        label_file:
        save_dir:
        device:

    Returns:

    """
    if isinstance(net_dirs, str):
        net_dirs = [net_dirs]

    eval_dir = os.path.dirname(image_file)
    image_name = os.path.basename(image_file)
    label_rel_name = os.path.relpath(label_file, eval_dir)
    data_handler = gn.DataHandler(
        eval_data_dir=eval_dir,
        eval_file=image_name,
        eval_label_file=label_rel_name)
    data_vol = data_handler.eval_volume

    image_cmaps = {'data': 'gray',
                   'segmentation': 'jet',
                   'prob_maps': 'pink'}
    save_name = 'prediction.tif'

    ndim_data = data_vol.ndim

    all_prob_maps = []
    for source in net_dirs:
        if isinstance(source, str):
            ckpt_dir = os.path.join(source, 'model', 'checkpoints', 'best')
            net = lab.restore_from_checkpoint(source, ckpt_dir, device)
        else:
            net = source

        print(f'Net input shape: {net.gene_graph.input_shape}. '
              f'Net output shape: {net.gene_graph.output_shape()}')

        output_shape = net.gene_graph.output_shape()
        net_is_3d = len(output_shape) == 3
        # Shape of the segmentation volume is same as data volume if
        # single-channel, else ignore the channel dimension
        if ndim_data > 3:
            vol_shape = data_vol.shape[1:]
        else:
            vol_shape = data_vol.shape
        # Initialize the volumes
        # segmentation = np.zeros(vol_shape)
        # Shape of the probability map volume: one map per class
        prob_shape = [data_handler.n_classes] + list(vol_shape)
        prob_maps = np.zeros(prob_shape)
        prob_map_update_dist = np.zeros(vol_shape, dtype=np.int)
        # Create an input_fn
        # Check if SAME padding is used
        gene0 = list(net.gene_graph.genes.items())[0][1]
        padding = gene0.hyperparam('padding_type')
        if padding.lower() == 'same':
            forward_window_overlap = [1] * (3 - len(output_shape)) + [s // 3 for s in output_shape]
        else:
            forward_window_overlap = [0] * 3

        predict_input_fn = data_handler.input_fn(
            mode=tf.estimator.ModeKeys.PREDICT,
            graph_source=net,
            forward_window_overlap=forward_window_overlap,
            prediction_volume=data_vol)
        # Inference pass result generator
        results: Iterator[Dict[str, np.ndarray]] = net.predict(predict_input_fn)

        # TODO: pass that distance scale triplet as a parameter instead of hard-coding
        if net_is_3d:
            distance_scale = (4, 1, 1)
        else:
            distance_scale = (1, 1)

        for r in results:
            patch_prob = r['probabilities']
            patch_dist = memoized_distance_transform(patch_prob.shape[1:], distance_scale)
            patch_corner = r['corner']
            if net_is_3d:
                z0 = patch_corner[0]
                z1 = z0 + output_shape[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[1]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[2]

                prev_update_dist = prob_map_update_dist[z0:z1, x0:x1, y0:y1]
                prev_prob = prob_maps[:, z0:z1, x0:x1, y0:y1]

                parts_to_update = prev_update_dist < patch_dist

                prev_prob[:, parts_to_update] = patch_prob[:, parts_to_update]
                prob_maps[:, z0:z1, x0:x1, y0:y1] = prev_prob

                prev_update_dist[parts_to_update] = patch_dist[parts_to_update]
                prob_map_update_dist[z0:z1, x0:x1, y0:y1] = prev_update_dist
            else:
                z0 = patch_corner[0]
                x0 = patch_corner[1]
                x1 = x0 + output_shape[0]
                y0 = patch_corner[2]
                y1 = y0 + output_shape[1]
                if ndim_data > 2:
                    prev_update_dist = prob_map_update_dist[z0, x0:x1, y0:y1]
                    prev_prob = prob_maps[:, z0, x0:x1, y0:y1]

                    parts_to_update = prev_update_dist < patch_dist

                    prev_prob[:, parts_to_update] = patch_prob[:, parts_to_update]
                    prob_maps[:, z0, x0:x1, y0:y1] = prev_prob

                    prev_update_dist[parts_to_update] = patch_dist[parts_to_update]
                    prob_map_update_dist[z0, x0:x1, y0:y1] = prev_update_dist
                else:
                    prev_update_dist = prob_map_update_dist[x0:x1, y0:y1]
                    prev_prob = prob_maps[:, x0:x1, y0:y1]

                    parts_to_update = prev_update_dist < patch_dist

                    prev_prob[:, parts_to_update] = patch_prob[:, parts_to_update]
                    prob_maps[:, x0:x1, y0:y1] = prev_prob

                    prev_update_dist[parts_to_update] = patch_dist[parts_to_update]
                    prob_map_update_dist[x0:x1, y0:y1] = prev_update_dist

        all_prob_maps.append(prob_maps)

    prob_map_mean = np.mean(all_prob_maps, axis=0)
    segmentation = np.argmax(prob_map_mean, axis=0)

    def tif_cmap(c):
        """Convert a matplotlib colormap into a tifffile colormap.

        """
        a = plt.get_cmap(c)(np.arange(256))
        return np.swapaxes(255 * a, 0, 1)[0:3, :].astype('u1')

    # Save a bunch of images, if `save_dir` was supplied
    if save_dir is not None:
        # Create a data volume image
        # For multichannel 3D data, only use the first channel, under the
        # assumption that that is actual image data
        # TODO: Find a more robust solution for multichannel data
        if ndim_data == 4:
            data_vol = data_vol[0, ...]
        # Generate a file name and path
        data_fname = f'train-data.tif'
        data_fpath = os.path.join(save_dir, data_fname)
        # Create a colormap compatible with tifffile's save function
        data_tcmap = tif_cmap(image_cmaps['data'])

        # Convert data volume to the right type
        data_image = (255. * (data_vol - data_vol.min()) /
                      (data_vol.max() - data_vol.min())).astype(np.uint8)
        # Save
        tif.imsave(data_fpath, data_image, colormap=data_tcmap)

        # Create a segmentation volume image
        # Generate a file name and path
        seg_fname = f'segmentation_{save_name}.tif'
        seg_fpath = os.path.join(save_dir, seg_fname)
        # Create a colormap compatible with tifffile's save function
        seg_tcmap = tif_cmap(image_cmaps['segmentation'])
        # Convert and scale the segmentation volume
        seg_image = (255. / (data_handler.n_classes - 1) *
                     segmentation).astype(np.uint8)
        # Save
        tif.imsave(seg_fpath, seg_image, colormap=seg_tcmap, compress=7)

    return segmentation


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


if __name__ == '__main__':
    main()