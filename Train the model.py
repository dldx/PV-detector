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

# ### Load the U-Net model and train it on our data!

# + {}
import sys
import os

# Add unet model folder
sys.path.append("unet/")

import config
from model import get_vgg_7conv

from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
import glob
from PIL import Image
import numpy as np


## Uncomment this if you want to force tensorflow to use your CPUs instead (useful if you don't have a beefy GPU)
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
# -

num_training_images = len(glob.glob(config.TRAIN_IMAGE_FOLDER + config.SAT_IMAGE_FOLDER + "/0/*.png"))
num_testing_images = len(glob.glob(config.TEST_IMAGE_FOLDER + config.SAT_IMAGE_FOLDER + "/0/*.png"))

if num_testing_images == 0 or num_training_images == 0:
    print("WARNING: no images detected")

def get_image_data(num_images_to_get=32, normalise=True, image_folder = config.TRAIN_IMAGE_FOLDER, augment=False):
    # Get list of files
    satellite_images = glob.glob(image_folder + config.SAT_IMAGE_FOLDER + "/0/*.png")
    if len(satellite_images) == 0:
        raise FileNotFoundError(image_folder + config.SAT_IMAGE_FOLDER + " doesn't contain any images!")
    if normalise:
        # first provide a sample of images for featurewise normalisation
        X_sample, y_sample = next(get_image_data(num_images_to_get=1000, normalise=False))

        mean_sat_color = np.mean(X_sample.reshape(X_sample.shape[0]*X_sample.shape[1]*X_sample.shape[2], 3), axis=0)
        variance_sat_color = np.var(X_sample.reshape(X_sample.shape[0]*X_sample.shape[1]*X_sample.shape[2], 3), axis=0)

        #mean_mask_color = np.mean(y_sample.reshape(y_sample.shape[0]*y_sample.shape[1]*y_sample.shape[2], 1), axis=0)
        #variance_mask_color = np.var(y_sample.reshape(y_sample.shape[0]*y_sample.shape[1]*y_sample.shape[2], 1), axis=0)
        
        del X_sample
        del y_sample

    while True:
        np.random.shuffle(satellite_images)

        # Preallocate array of image stacks
        satellite_stacked = np.zeros((num_images_to_get, 224, 224, 3), dtype=np.float32)
        mask_stacked = np.zeros((num_images_to_get, 224, 224, 1), dtype=np.float32)

        for i, image in enumerate(satellite_images[:num_images_to_get]):
            image = image.split("/")[-1]
            satellite_stacked[i] = np.array(Image.open(image_folder + config.SAT_IMAGE_FOLDER + "/0/" + image))
            mask_stacked[i] = np.array(Image.open(image_folder + config.MASK_IMAGE_FOLDER + "/0/" + image)).reshape(224,224, 1)/255.0

        if augment:
            augmenters = iaa.SomeOf((0, None), [
                iaa.WithChannels(0, iaa.Add((-20, 20))),
                iaa.WithChannels(1, iaa.Add((-20, 20))),
                iaa.WithChannels(2, iaa.Add((-20, 20))),
                iaa.Sharpen(alpha=0.1),
                iaa.GaussianBlur(sigma=(0, 2)),
                iaa.AdditiveGaussianNoise(scale=(0,0.05*255)),
            ])
            satellite_stacked = augmenters.augment_images(satellite_stacked)
        if normalise:
            satellite_stacked = (satellite_stacked - mean_sat_color)/np.sqrt(variance_sat_color)
            #mask_stacked = (mask_stacked - mean_mask_color)/np.sqrt(variance_mask_color)
        yield satellite_stacked, mask_stacked

# + {}
from keras import backend as K

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
# -

os.makedirs(config.WEIGHTS_FOLDER, exist_ok=True)
model_checkpoint = ModelCheckpoint(config.WEIGHTS_FOLDER + 'intermediate_models_with_aug.h5', monitor='dice_coef', save_best_only=True)
tb_callback = TensorBoard(log_dir=config.LOGS_FOLDER, histogram_freq=0, batch_size=config.BATCH_SIZE,
                          write_graph=True, write_grads=False, write_images=True,
                          embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# + {}
import metrics

model = get_vgg_7conv(config.ISZ, config.N_CHANNELS, config.NUM_CLASSES)

model.compile(
    optimizer=Adam(lr=0.001),
    loss=dice_coef_loss,
    metrics=[
        metrics.jaccard_coef, metrics.jacard_coef_flat,
        metrics.jaccard_coef_int, metrics.dice_coef, 'accuracy'
    ])
# -

model.fit_generator(
    get_image_data(num_images_to_get=config.BATCH_SIZE, image_folder=config.TRAIN_IMAGE_FOLDER),
    epochs=80, verbose=True,
    steps_per_epoch=int(num_training_images/config.BATCH_SIZE)//50*50,
    validation_data=get_image_data(num_images_to_get=config.BATCH_SIZE, image_folder=config.TEST_IMAGE_FOLDER),
    validation_steps=int(num_testing_images/config.BATCH_SIZE)//50*50,
    callbacks=[model_checkpoint,
               tb_callback,
               ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1, mode='max')])

model.save(WEIGHTS_FOLDER + "/model_with_aug.h5")
