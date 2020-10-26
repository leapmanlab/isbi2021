"""Touch up label files

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import remove_small_objects


def postprocess(label_file_out: str, label_file_in: str, min_size: int = 20):
    """Apply postprocessing to a label file. Currently just small region removal

    Args:
        label_file_out (str): Save path for the postprocessed label file.
        label_file_in (str): Path to the label file to be postprocessed.
        min_size (int): Smallest allowable object size.

    Returns: None

    """
    label_in = plt.imread(label_file_in)[..., 3] > 0
    label_out = remove_small_objects(label_in, min_size=min_size)
    layer_out = 255 * label_out.astype(np.uint8)
    arr_out = np.zeros((*label_out.shape, 4), dtype=np.uint8)
    arr_out[..., 2] = layer_out
    arr_out[..., 3] = layer_out
    plt.imsave(label_file_out, arr_out)
    pass


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    src_dir = args[1]
    min_size = 20 if argc < 3 else int(args[2])
    for f in [os.path.join(src_dir, _) for _ in os.listdir(src_dir)]:
        postprocess(f, f, min_size)
