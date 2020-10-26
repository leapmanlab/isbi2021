import os

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif

from scipy.ndimage.interpolation import zoom

from typing import Sequence

lcimb_dir = os.path.join('..', '..', '..', '..')


def main(label_files: Sequence[str], save_file: str):
    labels = np.stack([(plt.imread(f)[..., -1] > 0).astype(float)
                       for f in label_files], axis=0)
    labels = zoom(labels, (4, 1, 1), order=0)
    tif.imsave(
        save_file,
        255 * (labels > 0.5).astype(np.uint8),
        compress=7)
    pass


if __name__ == '__main__':
    label_dir = os.path.join(lcimb_dir, 'data', 'platelet-membrane', 'sample',
                             'train', 'label')
    label_files = [os.path.join(label_dir, f'{z:04}.png') for z in range(20)]
    save_file = os.path.join('.', 'media', 'writeup_1017', 'tomviz-bi4.tif')
    main(label_files, save_file)
