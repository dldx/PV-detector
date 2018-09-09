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

# import os
# os.chdir("./unet")
from model import get_unet
from config import *
from unet_config import Config
import os
from keras.callbacks import ModelCheckpoint, TensorBoard

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.preprocessing.image import ImageDataGenerator

import glob
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def get_image_data(sample=False):
    # Get list of files
    satellite_images = glob.glob(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/*.png")
    np.random.shuffle(satellite_images)
    if sample:
        num_images_to_get = max(1000, int(len(satellite_images)/100))
    else:
        num_images_to_get = len(satellite_images)

    # Preallocate array of image stacks
    satellite_stacked = np.zeros((num_images_to_get, 224, 224, 3), dtype=np.int32)
    mask_stacked = np.zeros((num_images_to_get, 224, 224, 1), dtype=np.int32)

    for i, image in enumerate(satellite_images[:num_images_to_get]):
        image = image.split("/")[-1]
        satellite_stacked[i] = np.array(Image.open(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER + "/0/" + image))
        mask_stacked[i] = np.array(Image.open(TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER + "/0/" + image)).reshape(224,224, 1)

    return satellite_stacked, mask_stacked

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True, featurewise_std_normalization=True, validation_split=0.2)
sat_image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
test_sat_image_datagen = ImageDataGenerator(**data_gen_args)
test_mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 42
#np.random.seed(seed)

# first provide a sample of images for featurewise normalisation
X_sample, y_sample = get_image_data(sample=True)
sat_image_datagen.fit(X_sample, augment=False, seed=seed)
mask_datagen.fit(y_sample, augment=False, seed=seed)

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

# combine generators into one which yields image and masks
train_generator = zip(sat_image_generator, mask_generator)

config_unet = Config()

os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
model_checkpoint = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER,'unet_2.hdf5'), monitor='loss', save_best_only=True)
tb_callback = TensorBoard(log_dir=LOGS_FOLDER, histogram_freq=0, batch_size=config_unet.BATCH_SIZE,
                          write_graph=True, write_grads=False, write_images=True,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# start_time = time.time()

model = get_unet(config_unet, loss_mode="tversky")
model.fit_generator(
    train_generator,
    epochs=config_unet.EPOCS, verbose=True,
    steps_per_epoch=200,
    shuffle=True,
    callbacks=[model_checkpoint, tb_callback])

## Testing the model
"""
config_unet = Config()
model = get_unet(config_unet, loss_mode="tversky")
model.load_weights(WEIGHTS_FOLDER + "/unet_tmp.hdf5")

sat, mask = get_image_data(sample=True)
plt.imshow(mask[0].reshape(224,224))
plt.imshow(model.predict(sat[0].reshape(1, 224,224,3)).reshape(224,224))
"""
