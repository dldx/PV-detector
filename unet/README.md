## Unet based model training and inference
Most of the code in this folder has been poached from [this repository by Olga Liakhovich](https://github.com/olgaliak/segmentation-unet-maskrcnn)

I will be all files that aren't required for the PV detection model shortly.

## Prerequisites
- Python 3.5+
- [Imgaug](https://github.com/aleju/imgaug)
- Tensorflow 1.6+
- Keras 2.0.8+

- unet_config.py

Confiuration for training and testing the model. Here is the place to provide file path to the data, images dimentions, number of epocs and etc.

- model.py

Here is definition of the Unet model.

- model_metrics.py

This file contains helper functions for model evaluation.

- losses.py

The place were custom loss functions (Dice coefficient based) reside.
