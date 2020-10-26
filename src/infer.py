import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import bio3d_vision as b3d
import comet_ml
import numpy as np
import torch
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import tifffile as tif

import torch.nn as nn
from models.unet import UNet
from models.deeplab import DeepLab
from utils import batch_generator, PlateletDataset, imshow, \
    memoized_distance_transform, stitch, evaluate
import json


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##################################################################
# Lines to change

# Name of network
experiment = "DeepLab"
backbone = "resnet"  # backbone for deeplab experiment

# Test or Eval dataset
test = True
test_cell = "3"
###################################################################
data_name = f"eval"
if test:
    data_name = f"test_{test_cell}"

# Fix the batch size to 1. Code will not run properly otherwise
BATCH_SIZE = 1

# Base directory for LCIMB projects
lcimb_dir = os.path.join('/home', 'zeyad', 'Documents')
eval_dataset_dir = os.path.join(lcimb_dir, 'data', 'platelet')

do_nonzero_masking = False
if test:
    eval_dataset_dir = os.path.join(lcimb_dir, 'data', 'platelet-donor2',
                                    test_cell)
    do_nonzero_masking = True

# Eval volumes
eval_data_file = 'eval-volume.tif'
eval_label_file = 'eval-label-volume.json'

if test:
    eval_data_file = 'raw.tif'
    eval_label_file = 'labels.tif'

# Number of classes in the data
n_classes = 7
if experiment == "Unet":
    windowing_params = {}
    windowing_params['random_windowing'] = False
    windowing_params['window_spacing'] = [1, 388, 388]
    windowing_params['image_window_shape'] = [1, 572, 572]
    windowing_params['label_window_shape'] = [1, 388, 388]
    model = UNet(n_classes=n_classes, padding=False, up_mode='upconv').to(
        device)
    channels = 1
    backbone = ""

elif experiment == "DeepLab":
    # Fix the windowing parameters
    windowing_params = dict()
    windowing_params['image_window_shape'] = [1, 572, 572]
    windowing_params['label_window_shape'] = [1, 572, 572]
    windowing_params['window_spacing'] = [1, 572, 572]
    windowing_params['random_windowing'] = False
    model = DeepLab(num_classes=n_classes,
                    backbone=backbone,
                    freeze_bn=False,
                    sync_bn=False)
    channels = 3

# Load the data
eval_image = b3d.load(eval_dataset_dir, eval_data_file)
eval_image = (eval_image-eval_image.min())*255/(eval_image.max() - eval_image.min())
eval_image -= np.mean(eval_image)
eval_image /= np.std(eval_image)

# Specify data type when loading label data
eval_label = b3d.load(eval_dataset_dir,
                      eval_label_file,
                      data_type=np.int32)

print(f"Shape of Image: {eval_image.shape}")
print(f"Shape of Labels: {eval_label.shape}")

ndim_data = eval_image.ndim

plot_settings = (
    {'cmap': 'gray'},
    {'cmap': 'jet', 'vmin': 0, 'vmax': eval_label.max()},
    {'cmap': 'jet', 'vmin': 0, 'vmax': eval_label.max()})

net_is_3d = False
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs.")

device_ids = [i for i in range(torch.cuda.device_count())]
model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

if experiment == "Unet":
    model.load_state_dict(torch.load("best_weights.pth"))

elif experiment == "DeepLab":
    model.load_state_dict(torch.load(f"best_weights_{backbone}_deeplab.pth"))

model.eval()

eval_images, eval_labels, eval_label_corners = batch_generator(
    eval_image,
    eval_label,
    **windowing_params,
    return_corners=True)

eval_dataset = PlateletDataset(eval_images, eval_labels, train=False)


prob_maps = stitch(model,
                   eval_images,
                   eval_labels,
                   eval_label.shape,
                   eval_label_corners,
                   windowing_params,
                   net_is_3d,
                   n_classes,
                   device,
                   channels)

stitched_classes = np.argmax(prob_maps, axis=0)

# A few plots for sanity check
for i in [0, 10, 20]:

    images = (np.squeeze(eval_image[i]),
        np.squeeze(stitched_classes[i]),
        np.squeeze(eval_label[i]))
    fig = imshow(images, (15, 5), plot_settings)
    plt.show()

eval_stats = evaluate(eval_label, stitched_classes, do_nonzero_masking)

print(f"{data_name} Stats {eval_stats}")

save_dir = os.path.join(".", f"{experiment}_{backbone}_segmentation",
                        data_name)
result_file = os.path.join(save_dir, 'stats.json')
data_fpath = os.path.join(save_dir)
os.makedirs(save_dir, exist_ok=True)

# Save stats
with open(result_file, 'w') as fd:
    json.dump(eval_stats, fd)
seg_image = (255. / (n_classes - 1) * stitched_classes).astype(np.uint8)

# Save segmented image
def tif_cmap(c):
    """Convert a matplotlib colormap into a tifffile colormap.

    """
    a = plt.get_cmap(c)(np.arange(256))
    return np.swapaxes(255 * a, 0, 1)[0:3, :].astype('u1')


seg_tcmap = tif_cmap('jet')
tif.imsave(data_fpath+".tif", seg_image, colormap=seg_tcmap, compress=7)
