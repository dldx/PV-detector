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

DATA_FOLDER = "/home/durand/Stuff/Data_Science/PV_detection/Data"

TILES_FOLDER = DATA_FOLDER + "/tiles/Only_PV_Tiles/"

SAT_IMAGE_FOLDER = "/satellites/"
MASK_IMAGE_FOLDER = "/masks/"

TRAIN_IMAGE_FOLDER = TILES_FOLDER + "/train/"
TEST_IMAGE_FOLDER = TILES_FOLDER + "/test/"
TRAIN_TEST_RATIO = 0.2

WEIGHTS_FOLDER = DATA_FOLDER + "/weights/"
LOGS_FOLDER = DATA_FOLDER + "/logs/"

# Image size for VGG16
TILE_SIZE = (224,224)
