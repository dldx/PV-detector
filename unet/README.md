## Unet based model training and inference
Most of the code in this folder has been poached from [this repository by Olga Liakhovich](https://github.com/olgaliak/segmentation-unet-maskrcnn)

I will be all files that aren't required for the PV detection model shortly.

## Prerequisites
- Python 3.5+
- Tensorflow 1.6+
- Keras 2.0.8+

- model.py

Here is definition of the Unet model.

- losses.py

The place were custom loss functions (Dice coefficient based) reside.
