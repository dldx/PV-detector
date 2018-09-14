# ---
# jupyter:
#   jupytext_format_version: '1.1'
#   jupytext_formats: ipynb,py
#   kernelspec:
#     display_name: Python [default]
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.6
# ---

# + {}
#import imgaug
import pandas as pd
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

from config import *
# -

# ### Load images

image_filenames = [file.replace(TILES_FOLDER + SAT_IMAGE_FOLDER, "") for file in glob.glob(TILES_FOLDER + SAT_IMAGE_FOLDER + "/*.png")]

# ### Split images into train and test

train_set, test_set = train_test_split(image_filenames, test_size=TRAIN_TEST_RATIO, random_state=42)
[os.makedirs(folder, exist_ok=True) for folder in [TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/", TEST_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/",
                                                 TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER  + "/0/", TEST_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/"]]


# + {}
# Move images into relevant directories
for image_name in train_set:
    shutil.move(TILES_FOLDER + SAT_IMAGE_FOLDER + image_name, TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/" + image_name)
    shutil.move(TILES_FOLDER + MASK_IMAGE_FOLDER + image_name, TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + image_name)
    
for image_name in test_set:
    shutil.move(TILES_FOLDER + SAT_IMAGE_FOLDER + image_name, TEST_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/" + image_name)
    shutil.move(TILES_FOLDER + MASK_IMAGE_FOLDER + image_name, TEST_IMAGE_FOLDER + MASK_IMAGE_FOLDER +  "/0/" + image_name)
    
# -

# ### Load tile details

tile_details

tile_details = pd.read_csv(DATA_FOLDER + "/tiles/tile_details.csv")

# Fraction of tiles that contain PVs

len(tile_details[tile_details.num_pv_pixels > 0])/len(tile_details)

tiles_with_PV = tile_details[tile_details.num_pv_pixels > 0].tile_image_names

# + {}
#shutil.rmtree(TILES_FOLDER + "/PV/test/" + SAT_IMAGE_FOLDER + "/0/")
#shutil.rmtree(TILES_FOLDER + "/PV/test/" + MASK_IMAGE_FOLDER + "/0/")

os.makedirs(TILES_FOLDER + "/PV/test/" + SAT_IMAGE_FOLDER + "/0/", exist_ok=True)
os.makedirs(TILES_FOLDER + "/PV/test/" + MASK_IMAGE_FOLDER + "/0/", exist_ok=True)
# -

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

for tile_image_name in tiles_with_PV:
    try:
        if os.path.exists(TEST_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + tile_image_name):
            shutil.copyfile(TEST_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/" + tile_image_name, TILES_FOLDER + "/PV/test/" + SAT_IMAGE_FOLDER + "/0/" + tile_image_name)
            im = np.array(Image.open(TEST_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + tile_image_name))
            im[im == 1] = 255
            Image.fromarray(im.astype('uint8')).save(TILES_FOLDER + "/PV/test/" + MASK_IMAGE_FOLDER + "/0/" + tile_image_name)
    except FileNotFoundError:
        pass

for tile_image_name in tiles_with_PV:
    try:
        if os.path.exists(TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + tile_image_name):
            shutil.copyfile(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/" + tile_image_name, TILES_FOLDER + "/PV/train/" + SAT_IMAGE_FOLDER + "/0/" + tile_image_name)
            im = np.array(Image.open(TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + tile_image_name))
            im[im == 1] = 255
            Image.fromarray(im.astype('uint8')).save(TILES_FOLDER + "/PV/train/" + MASK_IMAGE_FOLDER + "/0/" + tile_image_name)
    except FileNotFoundError:
        pass
