# ---
# jupyter:
#   jupytext_format_version: '1.1'
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

from model import get_unet
from config import *
from unet_config import Config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Test if on GPU or not (shows up in terminal)
import tensorflow as tf
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))


from keras.preprocessing.image import ImageDataGenerator

# def get_dataset(image_folder):
import glob
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_image_data(sample=False):
    # Get list of files
    satellite_images = glob.glob(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/*.png")
    np.random.shuffle(satellite_images)
    if sample:
        num_images_to_get = max(1000, len(satellite_images)/100)
    else:
        num_images_to_get = len(satellite_images)

    # Preallocate array of image stacks
    satellite_stacked = np.zeros((num_images_to_get, 224, 224, 3), dtype=np.int32)
    mask_stacked = np.zeros((num_images_to_get, 224, 224, 1), dtype=np.int32)

    for i, image in enumerate(satellite_images[:num_images_to_get]):
        image = image.split("/")[-1]
        satellite_stacked[i] = np.array(Image.open(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + image))
        mask_stacked[i] = np.array(Image.open(TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER + image)).reshape(224,224, 1)

    return satellite_stacked, mask_stacked

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True, featurewise_std_normalization=True)
sat_image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 42

# first provide a sample of images for featurewise normalisation
X_sample, y_sample = get_image_data(sample=True)
sat_image_datagen.fit(X_sample, augment=False, seed=seed)
mask_datagen.fit(y_sample, augment=False, seed=seed)
TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER
sat_image_generator = sat_image_datagen.flow_from_directory(
    TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER,
    target_size=TILE_SIZE,
    follow_links=True,
    color_mode="rgb",
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER,
    target_size=TILE_SIZE,
    follow_links=True,
    color_mode="grayscale",
    class_mode=None,
    seed=seed)

import itertools

# combine generators into one which yields image and masks
train_generator = itertools.islice(zip(sat_image_generator, mask_generator), 10)

model = get_unet(Config(), loss_mode="bce")
model.fit_generator(
    train_generator,
    steps_per_epoch=20,
    epochs=1, verbose=True)
