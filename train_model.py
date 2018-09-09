# ---
# jupyter:
#   jupytext_format_version: '1.1'
#   jupytext_formats: ipynb,py
#   kernelspec:
#     display_name: Python [conda env:PV_detection]
#     language: python
#     name: conda-env-PV_detection-py
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
import imgaug
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

from config import *
# -

# ### Load images

image_filenames = [file.replace(TILES_FOLDER + SAT_IMAGE_FOLDER, "") for file in glob.glob(TILES_FOLDER + SAT_IMAGE_FOLDER + "/*.png")]

TILES_FOLDER + SAT_IMAGE_FOLDER

# ### Split images into train and test

train_set, test_set = train_test_split(image_filenames, test_size=TRAIN_TEST_RATIO, random_state=42)
[os.makedirs(folder, exist_ok=True) for folder in [os.path.join(TRAIN_IMAGE_FOLDER + SAT_IMAGE_FOLDER), os.path.join(TEST_IMAGE_FOLDER + SAT_IMAGE_FOLDER),
                                                 os.path.join(TRAIN_IMAGE_FOLDER + MASK_IMAGE_FOLDER), os.path.join(TEST_IMAGE_FOLDER + MASK_IMAGE_FOLDER)]]


# + {}
# Move images into relevant directories
for image_name in train_set:
    try:
        shutil.move(os.path.join(TILES_FOLDER, SAT_IMAGE_FOLDER, image_name), os.path.join(TRAIN_IMAGE_FOLDER, SAT_IMAGE_FOLDER, image_name))
        shutil.move(os.path.join(TILES_FOLDER, MASK_IMAGE_FOLDER, image_name), os.path.join(TRAIN_IMAGE_FOLDER, MASK_IMAGE_FOLDER, image_name))
    except FileNotFoundError:
        pass

    
for image_name in test_set:
    try:
        shutil.move(os.path.join(TILES_FOLDER, SAT_IMAGE_FOLDER, image_name), os.path.join(TEST_IMAGE_FOLDER, SAT_IMAGE_FOLDER, image_name))
        shutil.move(os.path.join(TILES_FOLDER, MASK_IMAGE_FOLDER, image_name), os.path.join(TEST_IMAGE_FOLDER, MASK_IMAGE_FOLDER, image_name))
    except FileNotFoundError:
        pass
# -

# ### Augment training data


